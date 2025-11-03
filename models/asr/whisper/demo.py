#!/usr/bin/python3
# -*- coding: utf-8 -*-
import os
import time
import numpy as np
import argparse
from loguru import logger

import torch
from transformers import WhisperProcessor
from datasets import load_dataset

import tcim_lite as tcim

HOUMO_TARGET = os.getenv("HOUMO_TARGET")

def get_args() -> argparse.Namespace:
    """Parse commandline."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--processor_dir",
        dest="processor_dir",
        type=str,
        default="whisper-medium",
        help="processor dir",
    )
    parser.add_argument(
        "--encoder_path",
        dest="encoder_path",
        type=str,
        default=os.path.join("output", HOUMO_TARGET, "whisper_encoder.hmm"),
        help="houmo encoder model path",
    )
    parser.add_argument(
        "--decoder_path",
        dest="decoder_path",
        type=str,
        default=os.path.join("output", HOUMO_TARGET, "whisper_decoder.hmm"),
        help="houmo decoder model path",
    )
    parser.add_argument(
        "--prefill_path",
        dest="prefill_path",
        type=str,
        default=os.path.join("output", HOUMO_TARGET, "whisper_prefill.hmm"),
        help="houmo prefill model path",
    )
    args = parser.parse_args()
    return args


class HmWhisper:
    def __init__(self, encoder_path, decoder_path, prefill_path):
        super().__init__()
        weight_manager = tcim.runtime.WeightManager(0)
        option1 = tcim.runtime.Option(weight_manager)
        self.encoder = tcim.runtime.load(encoder_path, option=option1)
        logger.info("encoder model loaded")
        option2 = tcim.runtime.Option(weight_manager)
        self.decoder = tcim.runtime.load(decoder_path, option=option2)
        logger.info("decoder model loaded")
        option3 = tcim.runtime.Option(weight_manager)
        self.prefill = tcim.runtime.load(prefill_path, option=option3)
        logger.info("prefill model loaded")

    def run_encoder(self, input_features):
        input_features = input_features.numpy()
        self.encoder.set_input(self.encoder.get_input_name(0), input_features)
        self.encoder.run()
        self.encoder.sync()
        outputs = []
        for i in range(self.encoder.get_num_outputs()):
            outputs.append(
                torch.tensor(
                    self.encoder.get_output(self.encoder.get_output_name(i)).numpy()
                )
            )
        return outputs

    def run_decoder(self, inputs):
        for input_name, input_data in inputs.items():
            self.decoder.set_input(input_name, input_data.numpy())
        self.decoder.run()
        self.decoder.sync()
        outputs = []
        for i in range(self.decoder.get_num_outputs()):
            outputs.append(
                torch.tensor(
                    self.decoder.get_output(self.decoder.get_output_name(i)).numpy()
                )
            )
        return outputs

    def run_prefill(
        self,
        inputs,
    ):
        for input_name, input_data in inputs.items():
            self.prefill.set_input(input_name, input_data.numpy())
        self.prefill.run()
        self.prefill.sync()
        outputs = []
        for i in range(self.prefill.get_num_outputs()):
            outputs.append(
                torch.tensor(
                    self.prefill.get_output(self.prefill.get_output_name(i)).numpy()
                )
            )
        return outputs

    def get_input_names(self, model_type):
        if model_type == "encoder":
            self.model = self.encoder
        elif model_type == "decoder":
            self.model = self.decoder
        elif model_type == "prefill":
            self.model = self.prefill
        input_names = []
        for i in range(self.prefill.get_num_inputs()):
            input_names.append(self.prefill.get_input_name(i))
        return input_names


def asr(hmwhisper, processor, input_features):
    detect_ids = torch.tensor([[50258]])  # [1,1]
    default_decoder_ids = torch.tensor([[50258, 0, 50359, 50363]])  # [1,1]
    cache_position = torch.tensor([[0]])
    cache_position_prefill = torch.tensor([[0, 1, 2, 3]])

    # detect language  input_features  detect_ids => [1,51865]
    detect_encoder_out = hmwhisper.run_encoder(input_features.half())
    mask_atten = torch.ones(([1, 16, 1, 1024])).half()
    mask_atten[:, :, :, 0 + 1 :] *= -65504

    decoder_input_names = hmwhisper.get_input_names("decoder")
    decoder_detext_inputs = {
        decoder_input_names[0]: detect_ids.to(torch.int32),
        decoder_input_names[1]: cache_position.to(torch.int32),
        decoder_input_names[2]: torch.tensor([0]).to(torch.int32),
        decoder_input_names[3]: torch.tensor([1]).to(torch.int32),
        decoder_input_names[4]: mask_atten,
    }

    k_cache = [
        torch.ones([1, 16, 1024, 64], dtype=torch.float16) * (-65504) for i in range(24)
    ]
    v_cache = [
        torch.ones([1, 16, 1024, 64], dtype=torch.float16) * (-65504) for i in range(24)
    ]

    for data_detect, k_data_cache in zip(decoder_input_names[5:29], k_cache):
        decoder_detext_inputs[data_detect] = k_data_cache
    for data_detect, v_data_cache in zip(decoder_input_names[29:53], v_cache):
        decoder_detext_inputs[data_detect] = v_data_cache

    k_list = []
    for i in range(24):
        k_list.append(detect_encoder_out[2 * i])

    v_list = []
    for i in range(24):
        v_list.append(detect_encoder_out[2 * i + 1])

    for data_detect, k_data in zip(decoder_input_names[53:77], k_list):
        decoder_detext_inputs[data_detect] = k_data

    for data_detect, v_data in zip(decoder_input_names[77:101], v_list):
        decoder_detext_inputs[data_detect] = v_data

    output = hmwhisper.run_decoder(decoder_detext_inputs)
    logits, _, _ = output[0], output[1:25], output[25:49]

    lang_to_id = [
        50327,
        50334,
        50272,
        50350,
        50304,
        50355,
        50330,
        50292,
        50302,
        50347,
        50309,
        50315,
        50270,
        50283,
        50297,
        50285,
        50261,
        50281,
        50259,
        50262,
        50307,
        50310,
        50300,
        50277,
        50338,
        50265,
        50319,
        50333,
        50352,
        50354,
        50279,
        50276,
        50291,
        50339,
        50286,
        50312,
        50275,
        50311,
        50274,
        50266,
        50356,
        50329,
        50316,
        50323,
        50306,
        50264,
        50294,
        50345,
        50353,
        50336,
        50293,
        50301,
        50349,
        50295,
        50308,
        50296,
        50314,
        50320,
        50282,
        50343,
        50346,
        50313,
        50271,
        50342,
        50288,
        50328,
        50321,
        50269,
        50340,
        50267,
        50284,
        50263,
        50344,
        50332,
        50322,
        50298,
        50305,
        50324,
        50326,
        50317,
        50303,
        50357,
        50273,
        50318,
        50287,
        50299,
        50331,
        50289,
        50341,
        50348,
        50268,
        50351,
        50280,
        50290,
        50337,
        50278,
        50335,
        50325,
        50260,
    ]

    non_lang_mask = torch.ones_like(logits[0], dtype=torch.bool)
    non_lang_mask[0, list(lang_to_id)] = False
    logits[:, :, non_lang_mask[0]] = -np.inf
    lang_ids = logits.argmax(-1)

    non_lang_mask = torch.ones_like(logits, dtype=torch.bool)
    non_lang_mask[0, 0, list(lang_to_id)] = False
    logits[:, :, non_lang_mask[0][0]] = -np.inf
    lang_ids = logits.argmax(-1)

    default_decoder_ids[0, 1] = lang_ids  # [[50258, 50259, 50359, 50363]] # 34.5197

    mask_atten = torch.ones(([1, 16, 1, 1024])).half()
    mask_atten[:, :, :, 0 + 4 :] *= -65504

    prefill_input_names = hmwhisper.get_input_names("prefill")
    prefill_inputs = {
        prefill_input_names[0]: default_decoder_ids.to(torch.int32),
        prefill_input_names[1]: cache_position_prefill.to(torch.int32),
        prefill_input_names[2]: torch.tensor([0]).to(torch.int32),
        prefill_input_names[3]: torch.tensor([4]).to(torch.int32),
        prefill_input_names[4]: mask_atten,
    }

    for data_detect, k_data_cache in zip(prefill_input_names[5:29], k_cache):
        prefill_inputs[data_detect] = k_data_cache

    for data_detect, v_data_cache in zip(prefill_input_names[29:53], v_cache):
        prefill_inputs[data_detect] = v_data_cache

    for data_detect, k_data in zip(prefill_input_names[53:77], k_list):
        prefill_inputs[data_detect] = k_data

    for data_detect, v_data in zip(prefill_input_names[77:101], v_list):
        prefill_inputs[data_detect] = v_data

    output = hmwhisper.run_prefill(prefill_inputs)
    logits, new_k_cache, new_v_cache = output[0], output[1:25], output[25:49]
    next_token_logits = logits[:, -1, :].to(copy=True, dtype=torch.float32)
    next_tokens = torch.argmax(next_token_logits, dim=-1)
    default_decoder_ids = torch.cat([default_decoder_ids, next_tokens[:, None]], dim=-1)

    cnt = 3
    while default_decoder_ids.shape[1] < 448 and next_tokens.item() != 50257:
        cnt += 1

        mask_atten = torch.ones(([1, 16, 1, 1024])).half()
        mask_atten[:, :, :, cnt + 1 :] *= -65504

        prefill_inputs[prefill_input_names[0]] = next_tokens.unsqueeze(0).to(
            torch.int32
        )
        prefill_inputs[prefill_input_names[1]] = torch.tensor([[cnt]]).to(torch.int32)
        prefill_inputs[prefill_input_names[2]] = torch.tensor([cnt]).to(torch.int32)
        prefill_inputs[prefill_input_names[3]] = torch.tensor([1]).to(torch.int32)
        prefill_inputs[prefill_input_names[4]] = mask_atten

        for data_detect, k_data_cache in zip(prefill_input_names[5:29], new_k_cache):
            prefill_inputs[data_detect] = k_data_cache

        for data_detect, v_data_cache in zip(prefill_input_names[29:53], new_v_cache):
            prefill_inputs[data_detect] = v_data_cache
        output = hmwhisper.run_decoder(prefill_inputs)
        logits, new_k_cache, new_v_cache = output[0], output[1:25], output[25:49]
        next_token_logits = logits[:, -1, :].to(copy=True, dtype=torch.float32)
        next_tokens = torch.argmax(next_token_logits, dim=-1)
        default_decoder_ids = torch.cat(
            [default_decoder_ids, next_tokens[:, None]], dim=-1
        )

    transcription = processor.batch_decode(
        default_decoder_ids, skip_special_tokens=True
    )
    return transcription


if __name__ == "__main__":
    args = get_args()

    # init houmo whisper model
    if HOUMO_TARGET == "xh2":
        hmwhisper = HmWhisper(
            args.encoder_path,
            args.decoder_path,
            args.prefill_path,
        )
    else:
        raise ValueError("Unsupport houmo target!")

    processor = WhisperProcessor.from_pretrained(args.processor_dir)

    # load dataset
    ds = load_dataset(
        "hf-internal-testing/librispeech_asr_dummy", "clean", split="validation"
    )
    sample = ds[1]["audio"]
    input_features = processor(
        sample["array"], sampling_rate=sample["sampling_rate"], return_tensors="pt"
    ).input_features
    start_time = time.time()

    # run whisper model
    transcription = asr(hmwhisper, processor, input_features)
    total_time = time.time() - start_time

    logger.success("transcription:")
    print("\033[1;95m{}".format(transcription))
    logger.success(f"E2E Latency (End-to-End Latency): {total_time:.3f} seconds")
