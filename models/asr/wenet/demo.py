import argparse
import copy
import os
import sys

import numpy as np
import torch
import yaml
import torchaudio
import torchaudio.compliance.kaldi as kaldi
from loguru import logger

from wenet.utils.ctc_utils import get_blank_id
from wenet.utils.init_tokenizer import init_tokenizer

import tcim_lite as tcim

HOUMO_TARGET = os.getenv('HOUMO_TARGET', 'houmo')
MODELZOO_PATH = os.getenv('MODELZOO_PATH', '../../..')
INPUT_FILE = os.path.join(MODELZOO_PATH, 'data/audio/4s.wav')


from wenet.transformer.search import (
    ctc_greedy_search,
    ctc_prefix_beam_search,
    attention_beam_search,
    attention_rescoring,
    attention_rescoring_xh1,
    DecodeResult,
)


def cosine_similarity(x, y):
    dot_product = np.dot(x, y)
    norm_x = np.linalg.norm(x)
    norm_y = np.linalg.norm(y)
    return dot_product / (norm_x * norm_y)


def log_softmax(x, axis=-1):
    """
    使用numpy实现log_softmax函数

    参数：
    x：输入的numpy数组，可以是一维向量或者二维数组（二维数组每一行视为一个向量进行操作）
    axis：指定计算softmax的维度，默认为最后一维

    返回值：
    log_softmax的计算结果，与输入x的维度相同
    """
    # 计算softmax部分
    exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    sum_exp_x = np.sum(exp_x, axis=axis, keepdims=True)
    softmax_x = exp_x / sum_exp_x  
    # 对softmax结果取对数得到log_softmax
    #return softmax_x
    return np.log(softmax_x + 1e-5)


def resample(sample, resample_rate=16000):
    """Resample sample.
    Inplace operation.

    Args:
        sample: {key, wav, label, sample_rate}
        resample_rate: target resample rate

    Returns:
        {key, wav, label, sample_rate}
    """
    assert "sample_rate" in sample
    assert "wav" in sample
    sample_rate = sample["sample_rate"]
    waveform = sample["wav"]
    if sample_rate != resample_rate:
        sample["sample_rate"] = resample_rate
        sample["wav"] = torchaudio.transforms.Resample(
            orig_freq=sample_rate, new_freq=resample_rate
        )(waveform)
    return sample


def feature_extraction(
    waveform,
    num_mel_bins=80,
    frame_length=25,
    frame_shift=10,
    dither=0.0,
    sample_rate=16000,
    tgt_size=67,
):
    waveform = waveform * (1 << 15)
    feat = kaldi.fbank(
        waveform,
        num_mel_bins=num_mel_bins,
        frame_length=frame_length,
        frame_shift=frame_shift,
        dither=dither,
        energy_floor=0.0,
        sample_frequency=sample_rate,
    ).unsqueeze(0)
    feat_size = feat.shape[1]
    assert (
        feat_size <= tgt_size
    ), "feature size {} is larger than target size {}".format(feat_size, tgt_size)
    if feat_size < tgt_size:
        feat = torch.nn.functional.pad(feat, (0, 0, 0, tgt_size - feat.shape[1], 0, 0))
    return feat.float(), feat_size


def model_val(
        onnx_encoder_model,
        model_cfg,
        args,
        blank_id,
        tokenizer,
    ):  
    stride = model_cfg["stride"]
    tgt_size = 1500
    window_size = model_cfg["window_size"]
    window_shift = model_cfg["window_shift"]
    ck_len = ((stride - 1) * 4 + 7 - 1) * window_shift + window_size

    waveform, sample_rate = torchaudio.load(INPUT_FILE)
    assert sample_rate == 16000
    waveform_chunks = []

    if args.fast:
        waveform_chunks.append(waveform)
    else:
        for i in range(0, len(waveform[0]), ck_len):  # 685ms  equals a  chunk
            end = min(i + ck_len, len(waveform[0]))
            chunk = waveform[0, :end]
            if chunk.size(0) >= 400:
                waveform_chunks.append(chunk.unsqueeze(0))

    result = ""
    with torch.no_grad():
        for idx, ck in enumerate(waveform_chunks):
            ck_feat, ck_feat_length = feature_extraction(
                ck,
                num_mel_bins=80,
                frame_length=25,
                frame_shift=10,
                dither=0.0,
                sample_rate=16000,
                tgt_size=tgt_size,
            )
            # get mask cnn
            mask_cnn = torch.ones(1, tgt_size, dtype=torch.float)
            mask_cnn[:, ck_feat_length:] = 0
            mask_cnn = mask_cnn[:, 2::2][:, 2::2]
            mask_cnn = mask_cnn.unsqueeze(1)

            # get mask attn
            mask_attn = (1 - mask_cnn) * -128
            mask_attn = mask_attn.unsqueeze(1)
            input_info_i8 = onnx_encoder_model.get_input_info("input")
            input_info_f32 = input_info_i8.astype(np.float32)
            input_tensor_f32 = tcim.runtime.Tensor(input_info_f32, ck_feat.unsqueeze(0).numpy())
            input_tensor = tcim.runtime.Tensor(input_info_i8).to_host(to_contiguous=True)
            input_tensor_f32.cast_to(input_tensor)

            # set input
            onnx_encoder_model.set_input("input", input_tensor)
            onnx_encoder_model.set_input("mask_attn", mask_attn.numpy().astype(np.int8))
            onnx_encoder_model.set_input("mask_cnn", mask_cnn.numpy().astype(np.int8))

            # run & sync
            onnx_encoder_model.run()
            onnx_encoder_model.sync()

            # get output
            output_num = onnx_encoder_model.get_num_outputs()
            output_npy_predict = None
            for id in range(0, output_num):
                output_name = onnx_encoder_model.get_output_name(id)
                if output_name == "output":
                    output_npy_predict = onnx_encoder_model.get_output(output_name).cast(np.float32).numpy()

            ctc_output = log_softmax(output_npy_predict,-1)
            ctc_lens = mask_cnn.sum(-1).squeeze(0).int()

            if args.infer_mode == 0:
                ck_result = ctc_greedy_search(
                    torch.from_numpy(ctc_output), ctc_lens, blank_id
                )
            elif args.infer_mode == 1:
                ck_result = ctc_prefix_beam_search(
                    torch.from_numpy(ctc_output),
                    ctc_lens,
                    beam_size=args.beam_size,
                    blank_id=blank_id,
                )
            tokens = ck_result[0].tokens
            result = tokenizer.detokenize(tokens)[0]
    return result


def get_args():
    parser = argparse.ArgumentParser(description="recognize with your model")
    parser.add_argument("--model_path", type=str, default=1, help="model path")
    parser.add_argument("--config", type=str, default="train.yaml", help="config file")
    parser.add_argument("--test_data", default="data.list", help="test data file")
    parser.add_argument("--log_file", type=str, default="stream_val.log", help="log file")
    parser.add_argument("--pe_enc", type=str, default=None, help="pe encoder")
    parser.add_argument("--batch_size", type=int, default=1, help="asr result file")
    parser.add_argument(
        "--data_type",
        default="raw",
        choices=["raw", "shard"],
        help="train and cv data type",
    )
    parser.add_argument(
        "--beam_size", type=int, default=10, help="beam size for search"
    )
    parser.add_argument("--device", type=str, default="cuda", help="device")
    parser.add_argument("--verbose", action="store_true", help="verbose")
    parser.add_argument("--fast", action="store_true", help="fast")
    parser.add_argument("--use_chunk_model", action="store_true", help="use chunk")
    parser.add_argument("--calib_data_path", type=str, default=None, help="calib data")
    parser.add_argument("--infer_mode", type=int, default=1, help="infer mode")
    args = parser.parse_args()
    return args
    

if __name__ == "__main__":
    args = get_args()

    try:
        with open(args.config, "r") as fin:
            configs = yaml.load(fin, Loader=yaml.FullLoader)
    except FileNotFoundError:
        logger.error(f"Config file {args.config} not found.")
        exit(-1)
    test_conf = copy.deepcopy(configs["dataset_conf"])
    test_conf["filter_conf"]["max_length"] = 102400
    test_conf["filter_conf"]["min_length"] = 0
    test_conf["filter_conf"]["token_max_length"] = 102400
    test_conf["filter_conf"]["token_min_length"] = 0
    test_conf["filter_conf"]["max_output_input_ratio"] = 102400
    test_conf["filter_conf"]["min_output_input_ratio"] = 0
    test_conf["speed_perturb"] = False
    test_conf["spec_aug"] = False
    test_conf["spec_sub"] = False
    test_conf["spec_trim"] = False
    test_conf["shuffle"] = False
    test_conf["sort"] = False
    test_conf["cycle"] = 1
    test_conf["list_shuffle"] = False
    test_conf["fbank_conf"]["dither"] = 0.0
    test_conf["batch_conf"]["batch_type"] = "static"
    test_conf["batch_conf"]["batch_size"] = args.batch_size

    tokenizer = init_tokenizer(configs)
    _, blank_id = get_blank_id(configs, tokenizer.symbol_table)
    
    model_config = {
        "stride": 16,
        "window_size": 400,
        "window_shift": 160,
        "tgt_size": 1500
    }

    if args.calib_data_path:
        import shutil
        if os.path.exists(args.calib_data_path):
            shutil.rmtree(args.calib_data_path)
        os.makedirs(args.calib_data_path, exist_ok=True)

    model_path = os.path.join("output", HOUMO_TARGET, f"wenet_encoder.hmm")
    if not os.path.exists(model_path):
        print(f"Model file {model_path} not found.")
        exit(-1)
    module = tcim.runtime.load(model_path)

    logger.info(f"audio input file: {INPUT_FILE}")
    result = model_val(module, model_config, args, blank_id, tokenizer)
    logger.success(result)
