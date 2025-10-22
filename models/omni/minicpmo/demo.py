#!/usr/bin/python3
# -*- coding: utf-8 -*-
import os
import json
import time
import argparse
import re
import numpy as np
import torch
from loguru import logger
import torch.nn.functional as F
from transformers import (AutoTokenizer, 
                          AutoProcessor, 
                          BertTokenizerFast, 
                          TopPLogitsWarper, 
                          TopKLogitsWarper,
                          RepetitionPenaltyLogitsProcessor,
                          TemperatureLogitsWarper,
                          MaxLengthCriteria,
                          EosTokenCriteria,
                          LogitsProcessorList,
                          StoppingCriteriaList)
from PIL import Image
import tempfile
import math
from copy import deepcopy
import librosa
import soundfile as sf
import importlib.util
if not importlib.util.find_spec("sentence_transformers"):
    os.system("pip install sentence_transformers -i https://pypi.tuna.tsinghua.edu.cn/simple")
from sentence_transformers import util
from typing import Any, Optional, Tuple, Union, List
if not importlib.util.find_spec("vocos"):
    os.system("pip install vocos -i https://pypi.tuna.tsinghua.edu.cn/simple")
from vocos.spectral_ops import IMDCT, ISTFT
if not importlib.util.find_spec("moviepy"):
    os.system("pip install moviepy -i https://pypi.tuna.tsinghua.edu.cn/simple")
from moviepy import VideoFileClip

#from hmatc.utils import logger

import tcim_lite

HOUMO_TARGET = os.getenv('HOUMO_TARGET')
assert HOUMO_TARGET in ["xh2"], f"Unsupported HOUMO_TARGET: {HOUMO_TARGET}"
SUPPORTED_MODEL_TYPES = ["onnx", "houmo"]
EXAMPLES_MODE = {
    0: "all",   ## vision + audio + llm + tts
    1: "llm",   ## llm only
    2: "vllm",  ## vision + llm
    3: "mvllm", ## multi-vision + llm
}
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def get_args() -> argparse.Namespace:
    """Parse commandline."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tokenizer_dir",
        dest="tokenizer_dir",
        type=str,
        default="MiniCPM-o-2_6",
        help="tokenizer dir",
    )
    parser.add_argument(
        "--embedding_path",
        dest="embedding_path",
        type=str,
        default=os.path.join("output", HOUMO_TARGET, "hmquant"),
        help="houmo embedding weight path",
    )
    parser.add_argument(
        "--vit_path",
        dest="vit_path",
        type=str,
        default=os.path.join("output", HOUMO_TARGET, "minicpmo_visual.hmm"),
        help="houmo visual model path",
    )
    parser.add_argument(
        "--audio_path",
        dest="audio_path",
        type=str,
        default=os.path.join("output", HOUMO_TARGET, "minicpmo_audio.hmm"),
        help="houmo audio model path",
    )
    parser.add_argument(
        "--llm_prefill_path",
        dest="llm_prefill_path",
        type=str,
        default=os.path.join("output", HOUMO_TARGET, "minicpmo_llm_prefill.hmm"),
        help="houmo llm prefill model path",
    )
    parser.add_argument(
        "--llm_decode_path",
        dest="llm_decode_path",
        type=str,
        default=os.path.join("output", HOUMO_TARGET, "minicpmo_llm_decode.hmm"),
        help="houmo llm decode model path",
    )
    parser.add_argument(
        "--llm_projector_path",
        dest="llm_projector_path",
        type=str,
        default=os.path.join("output", HOUMO_TARGET, "minicpmo_llm_projector.hmm"),
        help="houmo llm projector model path",
    )
    parser.add_argument(
        "--tts_prefill_path",
        dest="tts_prefill_path",
        type=str,
        default=os.path.join("output", HOUMO_TARGET, "minicpmo_tts_prefill.hmm"),
        help="houmo tts Llama prefill model path",
    )
    parser.add_argument(
        "--tts_decode_path",
        dest="tts_decode_path",
        type=str,
        default=os.path.join("output", HOUMO_TARGET, "minicpmo_tts_decode.hmm"),
        help="houmo tts Llama decode model path",
    )
    parser.add_argument(
        "--tts_dvae_path",
        dest="tts_dave_path",
        type=str,
        default=os.path.join("output", HOUMO_TARGET, "minicpmo_dvae.hmm"),
        help="houmo tts dvae model path",
    )
    parser.add_argument(
        "--tts_vocos_path",
        dest="tts_vocos_path",
        type=str,
        default=os.path.join("output", HOUMO_TARGET, "minicpmo_vocos.hmm"),
        help="houmo tts vocos model path",
    )
    parser.add_argument(
        '--device_id',
        dest='device_id',
        type=int,
        default=0,
        help='Houmo device index',
    )
    parser.add_argument(
        '--example_idx',
        dest='example_idx',
        type=int,
        default=2,
        help='example mode index, support 0(all), 1(llm), 2(vllm), 3(mvllm)',
    )
    args = parser.parse_args()
    return args

def get_video_chunk_content(video_path, flatten=True):
    video = VideoFileClip(video_path)
    logger.info('video_duration:', video.duration)
    
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as temp_audio_file:
        temp_audio_file_path = temp_audio_file.name
        video.audio.write_audiofile(temp_audio_file_path, codec="pcm_s16le", fps=16000)
        audio_np, sr = librosa.load(temp_audio_file_path, sr=16000, mono=True)
    num_units = math.ceil(video.duration)
    
    # 1 frame + 1s audio chunk
    contents= []
    for i in range(num_units):
        frame = video.get_frame(i+1)
        image = Image.fromarray((frame).astype(np.uint8))
        audio = audio_np[sr*i:sr*(i+1)]
        if flatten:
            contents.extend(["<unit>", image, audio])
        else:
            contents.append(["<unit>", image, audio])
    
    return contents

def get_input_infos(engine):
    input_infos = {}
    for idx in range(engine.get_num_inputs()):
        input_name = engine.get_input_name(idx)
        input_infos[input_name] = engine.get_input_info(input_name)
    return input_infos

def get_output_infos(engine):
    output_infos = {}
    for idx in range(engine.get_num_outputs()):
        output_name = engine.get_output_name(idx)
        output_infos[output_name] = engine.get_output_info(output_name)
    return output_infos

def get_2d_sincos_pos_embed(embed_dim, image_size):
    """
    image_size: image_size or (image_height, image_width)
    return:
    pos_embed: [image_height, image_width, embed_dim]
    """
    if isinstance(image_size, int):
        grid_h_size, grid_w_size = image_size, image_size
    else:
        grid_h_size, grid_w_size = image_size[0], image_size[1]

    grid_h = np.arange(grid_h_size, dtype=np.float32)
    grid_w = np.arange(grid_w_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid_new(embed_dim // 2, grid[0])  # (H, W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid_new(embed_dim // 2, grid[1])  # (H, W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=-1)  # (H, W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid_new(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (H, W)
    out: (H, W, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    out = np.einsum("hw,d->hwd", pos, omega)  # (H, W, D/2), outer product

    emb_sin = np.sin(out)  # (H, W, D/2)
    emb_cos = np.cos(out)  # (H, W, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=-1)  # (H, W, D)
    return emb

def subsequent_chunk_mask(
    size: int,
    chunk_size: int,
    num_left_chunks: int = -1,
    device: torch.device = torch.device("cpu"),
    num_lookhead: int = 0,
) -> torch.Tensor:
    """Create mask for subsequent steps (size, size) with chunk size,
    this is for streaming encoder

    Args:
        size (int): size of mask
        chunk_size (int): size of chunk
        num_left_chunks (int): number of left chunks
            <0: use full chunk
            >=0: use num_left_chunks
        device (torch.device): "cpu" or "cuda" or torch.Tensor.device

    Returns:
        torch.Tensor: mask

    Examples:
        >>> subsequent_chunk_mask(4, 2)
        [[1, 1, 0, 0],
        [1, 1, 0, 0],
        [1, 1, 1, 1],
        [1, 1, 1, 1]]
    """
    ret = torch.zeros(size, size, device=device, dtype=torch.bool)
    for i in range(size):
        if num_left_chunks < 0:
            start = 0
        else:
            start = max((i // chunk_size - num_left_chunks) * chunk_size, 0)
        ending = min((i // chunk_size + 1) * chunk_size + num_lookhead, size)
        ret[i, start:ending] = True
    return ret

def _get_feat_extract_output_lengths(input_lengths: torch.LongTensor, audio_pool_step=2):
    """
    Computes the output length of the convolutional layers and the output length of the audio encoder
    """
    input_lengths_after_cnn = (input_lengths - 1) // 2 + 1
    input_lengths_after_pooling = (
        input_lengths_after_cnn - audio_pool_step
    ) // audio_pool_step + 1
    input_lengths_after_pooling = input_lengths_after_pooling.to(dtype=torch.int32)

    return input_lengths_after_cnn, input_lengths_after_pooling

class HMMiniCPMO(object):
    def __init__(self, 
                 args, 
                 init_vision=True,
                 init_audio=True,
                 init_tts=True,
                 use_tts_template=False, 
                 llm_sampling=True,
                 tts_sampling=True):
        self.init_vision = init_vision
        self.init_audio = init_audio
        self.init_tts = init_tts
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.use_tts_template = use_tts_template
        self.llm_sampling = llm_sampling
        self.max_position_embeddings = 32768
        self.vision_batch_size = 1
        self.default_tts_chat_template = "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n<|spk_bos|><|spk|><|spk_eos|><|tts_bos|>' }}{% endif %}"
        
        self.processer = AutoProcessor.from_pretrained(args.tokenizer_dir, trust_remote_code=True)

        self.embedding_path = os.path.join(args.embedding_path, "quant_embedding.pt")
        self.embedding = torch.load(self.embedding_path, map_location=torch.device(self.device), weights_only=False)
        self.hidden_dims = self.embedding.shape[-1]
        self.tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_dir, trust_remote_code=True)

        wt_manager = tcim_lite.runtime.WeightManager(args.device_id)
        
        if self.init_vision:
            option_vpm = tcim_lite.runtime.Option(wt_manager)
            self.vpm_engine = tcim_lite.runtime.load(args.vit_path, option_vpm)
            self.vpm_input_infos = get_input_infos(self.vpm_engine)
            self.vpm_output_infos = get_output_infos(self.vpm_engine)
            self.vpm_patch_size = 14
            self.vpm_max_size = (70, 70)
            self.vpm_pos_embed = torch.from_numpy(get_2d_sincos_pos_embed(self.vpm_output_infos["hidden_state"].shape[-1], self.vpm_max_size)).float().to(self.device)
        
        if self.init_audio:
            option_apm = tcim_lite.runtime.Option(wt_manager)
            self.apm_engine = tcim_lite.runtime.load(args.audio_path, option_apm)
            self.apm_input_infos = get_input_infos(self.apm_engine)
            self.apm_output_infos = get_output_infos(self.apm_engine)

        self.llm_temperature = 0.5
        self.llm_top_k = 100
        self.llm_top_p = 0.8
        self.llm_min_tokens_to_keep = 1
        self.llm_filter_value = -float("Inf")
        self.llm_penalty = 1.05
        self.eos_token_id = [151645, 151643]
        
        option_llm_prefill = tcim_lite.runtime.Option(wt_manager)
        self.llm_prefill_engine = tcim_lite.runtime.load(args.llm_prefill_path, option_llm_prefill)
        self.llm_prefill_input_infos = get_input_infos(self.llm_prefill_engine)
        self.llm_prefill_output_infos = get_output_infos(self.llm_prefill_engine)
        self.llm_nblocks = self.get_nblocks(self.llm_prefill_engine)
        self.llm_context_max_length = self.llm_prefill_input_infos[self.llm_prefill_engine.get_input_name(3)].shape[2]
        llm_prefill_shape = self.llm_prefill_input_infos[self.llm_prefill_engine.get_input_name(0)].shape[:2]
        self.llm_prefill_shape = torch.Size(llm_prefill_shape)
        self.llm_prefill_len = self.llm_prefill_shape.numel()
        self.llm_output_names = sorted(list(self.llm_prefill_output_infos.keys()), reverse=True)

        self.llm_prepared_logits_processer = LogitsProcessorList()
        self.llm_prepared_logits_processer.append(RepetitionPenaltyLogitsProcessor(self.llm_penalty))
        if self.llm_sampling:
            self.llm_prepared_logits_warper = LogitsProcessorList()
            for w in [TemperatureLogitsWarper(self.llm_temperature), 
                    TopKLogitsWarper(self.llm_top_k, self.llm_min_tokens_to_keep),
                    TopPLogitsWarper(self.llm_top_p, self.llm_min_tokens_to_keep)]:
                self.llm_prepared_logits_warper.append(w)
            self.llm_parpared_stopping_criteria = StoppingCriteriaList()
            for c in [MaxLengthCriteria(self.llm_context_max_length, self.max_position_embeddings),
                    EosTokenCriteria(torch.tensor(self.eos_token_id, dtype=torch.long, device=self.device))]:
                self.llm_parpared_stopping_criteria.append(c)

        option_llm_decoder = tcim_lite.runtime.Option(wt_manager)
        dummy_llm_tensor_names = [f'model_layers_{i}_self_attn_kcache_input' for i in range(self.llm_nblocks)]
        dummy_llm_tensor_names += [f'model_layers_{i}_self_attn_vcache_input' for i in range(self.llm_nblocks)]
        option_llm_decoder.set_dummy_tensors(dummy_llm_tensor_names)
        self.llm_decoder_engine = tcim_lite.runtime.load(args.llm_decode_path, option_llm_decoder)
        self.llm_decoder_input_infos = get_input_infos(self.llm_decoder_engine)
        self.llm_decoder_output_infos = get_output_infos(self.llm_decoder_engine)
        for i in range(self.llm_nblocks):
            kcache = self.llm_prefill_engine.get_input(f"model_layers_{i}_self_attn_kcache_input")
            vcache = self.llm_prefill_engine.get_input(f"model_layers_{i}_self_attn_vcache_input")
            self.llm_decoder_engine.set_input(f"model_layers_{i}_self_attn_kcache_input", kcache)
            self.llm_decoder_engine.set_input(f"model_layers_{i}_self_attn_vcache_input", vcache)
        self.llm_decoder_engine.set_input("current_length", np.array([1]).astype(self.llm_decoder_input_infos["current_length"].dtype))

        if self.init_tts:
            self.tts_text_tokenizer = BertTokenizerFast.from_pretrained(f"{args.tokenizer_dir}/assets/chattts_tokenizer")
            self.tts_streaming_text_reserved_len = 300
            self.tts_streaming_audio_chunk_size = 50
            self.tts_num_spk_embs = 1
            self.tts_use_speaker_embedding = True
            self.tts_top_p = 0.7
            self.tts_top_k = 20
            self.tts_repetition_penalty = 1
            self.tts_num_vq = 4
            self.tts_streaming_text_chunk_size = 10
            self.tts_spk_emb_token_id = 21143
            self.tts_audio_bos_token_id = 21132
            self.tts_temperature = torch.tensor([0.1, 0.3, 0.1, 0.3], dtype=torch.float, device=self.device)
            self.tts_max_new_token = 25
            self.tts_min_new_token = 10
            self.tts_eos_token = torch.tensor([625], dtype=torch.long, device=self.device)
            self.tts_sampling = tts_sampling
            if self.tts_sampling:
                self.tts_logits_warpers = [TopPLogitsWarper(self.tts_top_p, mini_tokens_to_keep=3),
                                TopKLogitsWarper(self.tts_top_k, mini_tokens_to_keep=3)]

            self.tts_embedding_path = os.path.join(args.embedding_path, "quant_embedding_tts.pt")
            self.tts_embedding = torch.load(self.tts_embedding_path, weights_only=False).weight
            self.tts_code_embeddings = self.load_tts_code_embeds(args.embedding_path)

            option_tts_projector = tcim_lite.runtime.Option(wt_manager)
            self.tts_projector_engine = tcim_lite.runtime.load(args.llm_projector_path, option_tts_projector)
            self.tts_projector_input_infos = get_input_infos(self.tts_projector_engine)
            self.tts_projector_output_infos = get_output_infos(self.tts_projector_engine)

            option_tts_prefill = tcim_lite.runtime.Option(wt_manager)
            self.tts_prefill_engine = tcim_lite.runtime.load(args.tts_prefill_path, option_tts_prefill)
            self.tts_prefill_input_infos = get_input_infos(self.tts_prefill_engine)
            self.tts_prefill_output_infos = get_output_infos(self.tts_prefill_engine)
            self.tts_nblocks = self.get_nblocks(self.tts_prefill_engine)
            tts_prefill_shape = self.tts_prefill_input_infos[self.tts_prefill_engine.get_input_name(0)].shape
            self.tts_prefill_shape = torch.Size(tts_prefill_shape)
            self.tts_context_max_length = self.tts_prefill_input_infos[self.tts_prefill_engine.get_input_name(4)].shape[2]
            self.tts_num_attention_heads = self.tts_prefill_shape[1]
            self.tts_hidden_size = self.tts_prefill_shape[-1]
            self.tts_mask_shape = self.tts_prefill_output_infos["attention_mask"].shape
            self.tts_mask_dtype=self.tts_prefill_output_infos["attention_mask"].dtype

            option_tts_decoder = tcim_lite.runtime.Option(wt_manager)
            self.tts_decoder_engine = tcim_lite.runtime.load(args.tts_decode_path, option_tts_decoder)
            self.tts_decoder_input_infos = get_input_infos(self.tts_decoder_engine)
            self.tts_decoder_output_infos = get_output_infos(self.tts_decoder_engine)
            for i in range(self.tts_nblocks):
                kcache = self.tts_prefill_engine.get_input(f"model_layers_{i}_self_attn_kcache_input")
                vcache = self.tts_prefill_engine.get_input(f"model_layers_{i}_self_attn_vcache_input")
                self.tts_decoder_engine.set_input(f"model_layers_{i}_self_attn_kcache_input", kcache)
                self.tts_decoder_engine.set_input(f"model_layers_{i}_self_attn_vcache_input", vcache)
            self.tts_decoder_engine.set_input("current_length", np.array([1]).astype(self.tts_decoder_input_infos["current_length"].dtype))

            option_dvae = tcim_lite.runtime.Option(wt_manager)
            self.dvae_engine = tcim_lite.runtime.load(args.tts_dvae_path, option_dvae)
            self.dvae_input_infos = get_input_infos(self.dvae_engine)
            self.dvae_output_infos = get_output_infos(self.dvae_engine)

            option_vocos = tcim_lite.runtime.Option(wt_manager)
            self.vocos_engine = tcim_lite.runtime.load(args.tts_vocos_path, option_vocos)
            self.vocos_input_infos = get_input_infos(self.vocos_engine)
            self.vocos_output_infos = get_output_infos(self.vocos_engine)
            self.vocos_istft = ISTFT(n_fft=1024, hop_length=256, win_length=1024, padding="same")
            self.wav_sr = 24000
        
        self.vision_time = 0.0
        self.audio_time = 0.0
        self.llm_prefill_time = 0.0
        self.llm_decode_time = 0.0
        self.tts_time = 0.0
        self.llm_ttft_start_time = 0.0
        self.llm_ttft_time = 0.0

    def load_tts_code_embeds(self, path_str):
        x = []
        for i in range(self.tts_num_vq):
            index_emb_path = os.path.join(path_str, f"quant_embedding_tts_code_{i}.pt")
            if not os.path.exists(index_emb_path):
                logger.error(f"{index_emb_path} is not exist! Please check it.")
                assert(0)
            x.append(torch.load(index_emb_path, weights_only=False).weight)
        return x

    def get_nblocks(self, engine):
        input_names = []
        for i in range(engine.get_num_inputs()):
            input_names.append(engine.get_input_name(i))
        pattern = r'^model_layers_(\d+)_self_attn_kcache_input$'
        count = sum(1 for item in input_names if re.match(pattern, item))
        return count

    def regularize_msgs(self, msgs_list, max_slice_nums, omni_input=False):
        prompts_lists = []
        input_images_list = []
        input_audios_list = []
        audio_parts_list = []
        for msgs in msgs_list:
            if isinstance(msgs, str):
                msgs = json.loads(msgs)
            copy_msgs = deepcopy(msgs)
            assert len(msgs) > 0, "msgs is empty"

            images = []
            audios = []
            audio_parts = []
            for i, msg in enumerate(copy_msgs):
                role = msg["role"]
                content = msg["content"]
                assert role in ["system", "user", "assistant"]
                if i == 0:
                    assert role in ["user", "system"], "The role of first msg should be user"
                if isinstance(content, str):
                    content = [content]
                cur_msgs = []
                for c in content:
                    if isinstance(c, Image.Image):
                        images.append(c)
                        cur_msgs.append("(<image>./</image>)")
                    elif isinstance(c, np.ndarray):  # audio
                        audios.append(c)
                        audio_parts.append(i)
                        cur_msgs.append("(<audio>./</audio>)")
                        self.use_tts_template = True
                    elif isinstance(c, str):
                        cur_msgs.append(c)
                if omni_input:
                    msg["content"] = "".join(cur_msgs)
                else:
                    msg["content"] = "\n".join(cur_msgs)
            prompts_lists.append(
                self.processer.tokenizer.apply_chat_template(
                    copy_msgs,
                    tokenize=False,
                    add_generation_prompt=True,
                    chat_template=self.default_tts_chat_template if self.use_tts_template else None,
                )
            )
            input_images_list.append(images)
            input_audios_list.append(audios)
            audio_parts_list.append(audio_parts)
        inputs = self.processer(prompts_lists,
                                input_images_list,
                                input_audios_list,
                                audio_parts_list,
                                max_slice_nums=max_slice_nums,
                                use_image_id=False,
                                chunk_input=True,
                                return_tensors="pt",
                                max_length=self.max_position_embeddings).to(self.device)
        return inputs
    
    def _adjust_post_cache(self, tgt_sizes):
        max_h = torch.max(tgt_sizes[:, 0])
        max_w = torch.max(tgt_sizes[:, 1])
        if max_h > self.vpm_max_size[0] or max_w > self.vpm_max_size[1]:
            self.vpm_max_size = (max(max_h, self.vpm_max_size[0]), max(max_w, self.vpm_max_size[1]))
            self.vpm_pos_embed = torch.from_numpy(get_2d_sincos_pos_embed(self.vpm_output_infos["hidden_state"].shape[-1], self.vpm_max_size).float().to(self.device))

    def get_vpm_position_ids(self, 
                             position_ids:torch.LongTensor,  
                             patch_attention_mask:torch.BoolTensor, 
                             tgt_sizes: Optional[torch.IntTensor]):
        boundaries = torch.arange(1 / self.vpm_max_size[0], 1.0, 1 / self.vpm_max_size[0])
        for batch_idx, p_attn_mask in enumerate(patch_attention_mask):
            if tgt_sizes is not None:
                nb_patches_h = tgt_sizes[batch_idx][0]
                nb_patches_w = tgt_sizes[batch_idx][1]
            else:
                nb_patches_h = p_attn_mask[:, 0].sum()
                nb_patches_w = p_attn_mask[0].sum()
            fractional_coords_h = torch.arange(0, 1 - 1e-6, 1 / nb_patches_h)
            fractional_coords_w = torch.arange(0, 1 - 1e-6, 1 / nb_patches_w)

            bucket_coords_h = torch.bucketize(fractional_coords_h, boundaries, right=True)
            bucket_coords_w = torch.bucketize(fractional_coords_w, boundaries, right=True)
            pos_ids = (bucket_coords_h[:, None] * self.vpm_max_size[0] + bucket_coords_w).flatten()
            position_ids[batch_idx][p_attn_mask.view(-1).cpu()] = pos_ids
        position_ids = position_ids.to(self.device)
        return position_ids

    def chat_vpm(self, data):
        if "vision_hidden_states" not in data:
            tgt_sizes = data["tgt_sizes"]
            pixel_values_list = data["pixel_values"]
            vision_hidden_states = []
            all_pixel_values = []
            img_cnt = []
            for pixel_values in pixel_values_list:
                img_cnt.append(len(pixel_values))
                all_pixel_values.extend([i.flatten(end_dim=1).permute(1, 0) for i in pixel_values])
            
            # exist image
            if all_pixel_values and self.init_vision:
                tgt_sizes = [tgt_size for tgt_size in tgt_sizes if isinstance(tgt_size, torch.Tensor)]
                tgt_sizes = torch.vstack(tgt_sizes).type(torch.int32)

                max_patches = torch.max(tgt_sizes[:, 0] * tgt_sizes[:, 1])

                all_pixel_values = torch.nn.utils.rnn.pad_sequence(
                    all_pixel_values, batch_first=True, padding_value=0.0
                )
                B, L, _ = all_pixel_values.shape
                all_pixel_values = all_pixel_values.permute(0, 2, 1).reshape(B, 3, -1, L)

                patch_len = tgt_sizes[:, 0] * tgt_sizes[:, 1]
                self._adjust_post_cache(tgt_sizes)
                max_patch_len = torch.max(patch_len)
                key_padding_mask = torch.zeros((B, max_patch_len), dtype=torch.bool, device=self.device)

                patch_attn_mask = torch.zeros((B, 1, max_patches), dtype=torch.bool, device=self.device)
                for i in range(B):
                    patch_attn_mask[i, 0, : tgt_sizes[i][0] * tgt_sizes[i][1]] = True

                vision_batch_size = self.vision_batch_size
                all_pixel_values = all_pixel_values.type(torch.float16)
                if B > vision_batch_size:
                    vision_embed_list = []
                    for i in range(0, B, vision_batch_size):
                        start_idx = i
                        end_idx = i + vision_batch_size

                        truth_pixel_value = all_pixel_values[start_idx:end_idx]
                        position_ids = torch.full(size=self.vpm_input_infos["position_ids"].shape, fill_value=0,)
                        attn_mask:torch.Tensor = patch_attn_mask[start_idx:end_idx]
                        truth_position_ids = self.get_vpm_position_ids(position_ids, attn_mask, tgt_sizes[start_idx:end_idx])
                        patch_attn_mask = attn_mask.view(1, -1)
                        if not torch.any(~patch_attn_mask):
                            truth_attn_mask = torch.zeros(self.vpm_input_infos["attention_mask"].shape, dtype=torch.float16)
                        else:
                            from transformers.modeling_attn_mask_utils import _prepare_4d_attention_mask
                            truth_attn_mask = _prepare_4d_attention_mask(patch_attn_mask, dtype=torch.float16)
                        tgt_h, tgt_w = tgt_sizes[i]
                        truth_pos_embed:torch.Tensor = self.vpm_pos_embed[:tgt_h, :tgt_w, :].reshape((tgt_h * tgt_w, -1)).unsqueeze(0).permute(1, 0, 2)
                        key_padding_mask[i, patch_len[i]:] = True
                        use_key_padding_mask = key_padding_mask[start_idx:end_idx, :self.vpm_input_infos["resampler_key_padding_mask"].shape[1]]
                        truth_key_padding_mask = torch.zeros_like(use_key_padding_mask, 
                                                                  dtype=torch.float16).masked_fill_(use_key_padding_mask, float("-inf"))
                        truth_pixel_value = truth_pixel_value.cpu().numpy().astype(self.vpm_input_infos["pixel_values"].dtype)
                        truth_position_ids = truth_position_ids.cpu().numpy().astype(self.vpm_input_infos["position_ids"].dtype)
                        truth_attn_mask = truth_attn_mask.cpu().numpy().astype(self.vpm_input_infos["attention_mask"].dtype)
                        truth_pos_embed = truth_pos_embed.cpu().numpy().astype(self.vpm_input_infos["resampler_pos_embed"].dtype)
                        truth_key_padding_mask = truth_key_padding_mask.cpu().numpy().astype(self.vpm_input_infos["resampler_key_padding_mask"].dtype)

                        self.vpm_engine.set_input("pixel_values", truth_pixel_value)
                        self.vpm_engine.set_input("position_ids", truth_position_ids)
                        self.vpm_engine.set_input("attention_mask", truth_attn_mask)
                        self.vpm_engine.set_input("resampler_pos_embed", truth_pos_embed)
                        self.vpm_engine.set_input("resampler_key_padding_mask", truth_key_padding_mask)

                        self.vpm_engine.run()
                        self.vpm_engine.sync()

                        sg_vision_embeding = torch.from_numpy(self.vpm_engine.get_output(list(self.vpm_output_infos.keys())[0]).numpy()).to(self.device)
                        vision_embed_list.append(sg_vision_embeding)
                    vision_embeding = torch.concat(vision_embed_list, dim=0)
                else:
                    truth_pixel_value = all_pixel_values
                    truth_position_ids = torch.full(size=self.vpm_input_infos["position_ids"].shape, fill_value=0,)
                    truth_attn_mask:torch.Tensor = patch_attn_mask
                    tgt_h, tgt_w = tgt_sizes[i]
                    truth_pos_embed:torch.Tensor = self.vpm_pos_embed[:tgt_h, :tgt_w, :].reshape((tgt_h * tgt_w, -1)).permute(1, 0, 2)
                    for i in range(B):
                        key_padding_mask[i, patch_len[i], :] = True
                    truth_key_padding_mask = torch.zeros_like(key_padding_mask, 
                                                                dtype=torch.float16).masked_fill_(key_padding_mask, float("-inf"))
                    truth_pixel_value = truth_pixel_value.cpu().numpy().astype(self.vpm_input_infos["pixel_values"].dtype)
                    truth_position_ids = truth_position_ids.cpu().numpy().astype(self.vpm_input_infos["position_ids"].dtype)
                    truth_attn_mask = truth_attn_mask.cpu().numpy().astype(self.vpm_input_infos["attention_mask"].dtype)
                    truth_pos_embed = truth_pos_embed.cpu().numpy().astype(self.vpm_input_infos["resampler_pos_embed"].dtype)
                    truth_key_padding_mask = truth_key_padding_mask.cpu().numpy().astype(self.vpm_input_infos["resampler_key_padding_mask"])

                    self.vpm_engine.set_input("pixel_values", truth_pixel_value)
                    self.vpm_engine.set_input("position_ids", truth_position_ids)
                    self.vpm_engine.set_input("attention_mask", truth_attn_mask)
                    self.vpm_engine.set_input("resampler_pos_embed", truth_pos_embed)
                    self.vpm_engine.set_input("resampler_key_padding_mask", truth_key_padding_mask)

                    self.vpm_engine.run()
                    self.vpm_engine.sync()

                    vision_embeding = torch.from_numpy(self.vpm_engine.get_output(list(self.vpm_output_infos.keys())[0])).to(self.device)
                start = 0
                for pixel_values in pixel_values_list:
                    img_cnt = len(pixel_values)
                    if img_cnt > 0:
                        vision_hidden_states.append(vision_embeding[start:start+img_cnt])
                        start += img_cnt
                    else:
                        vision_hidden_states.append([])
            else:
                if not self.init_vision:
                    logger.warning("Vision encoder model is not initialized! But there is image input! Will skip image input!")
                for _ in range(len(pixel_values_list)):
                    vision_hidden_states.append([])
        else:
            vision_hidden_states = data["vision_hidden_states"]
        return vision_hidden_states
    
    def chat_audio(self, data):
        wavforms = data.get("audio_features", [])
        audio_feature_lens_raw = data.get("audio_feature_lens", [])

        if self.init_audio:
            logger.warning("Audio encoder model is not initialized! But there is audio input! Will skip audio input!")
            return []
        # exist audio
        if len(wavforms) > 0:
            audio_feature_lens = torch.hstack(audio_feature_lens_raw)
            batch_size, _, max_mel_seq_len = wavforms.shape
            max_seq_len = (max_mel_seq_len - 1) // 2 + 1

            # Create a sequence tensor of shape (batch_size, max_seq_len)
            seq_range = (
                torch.arange(0, max_seq_len, dtype=audio_feature_lens.dtype, device=audio_feature_lens.device)
                .unsqueeze(0)
                .expand(batch_size, max_seq_len)
            )
            lengths_expand = audio_feature_lens.unsqueeze(1).expand(batch_size, max_seq_len)
            # Create mask
            padding_mask = seq_range >= lengths_expand  # 1 for padded values

            audio_attention_mask_ = padding_mask.view(batch_size, 1, 1, max_seq_len).expand(
                batch_size, 1, max_seq_len, max_seq_len
            )
            audio_attention_mask = audio_attention_mask_.to(dtype=torch.float16, device=self.device)

            chunk_mask = subsequent_chunk_mask(
                size=max_seq_len,
                chunk_size=50,
                num_left_chunks=-1,
                device=audio_attention_mask_.device,
            )
            audio_attention_mask_ = torch.logical_or(audio_attention_mask_, torch.logical_not(chunk_mask))
            audio_attention_mask[audio_attention_mask_] = float("-inf")

            truth_input_features = wavforms.detach().cpu().numpy().astype(self.apm_input_infos["input_features"].dtype)
            truth_audio_attention_mask = audio_attention_mask.detach().cpu().numpy().astype(self.apm_input_infos["audio_attention_mask"].dtype)

            self.apm_engine.set_input("input_features", truth_input_features)
            self.apm_engine.set_input("audio_attention_mask", truth_audio_attention_mask)

            self.apm_engine.run()
            self.apm_engine.sync()

            audio_embeds = torch.from_numpy(self.apm_engine.get_output(list(self.apm_output_infos.keys())[0])).to(self.device)

            _, feature_lens_after_pooling = _get_feat_extract_output_lengths(audio_feature_lens)

            num_audio_tokens = feature_lens_after_pooling

            final_audio_embeds = []
            idx = 0
            for i in range(len(audio_feature_lens_raw)):
                target_audio_embeds = []
                for _ in range(len(audio_feature_lens_raw[i])):
                    target_audio_embeds.append(audio_embeds[idx, : num_audio_tokens[idx], :])
                    idx += 1
                final_audio_embeds.append(target_audio_embeds)
            return final_audio_embeds
        else:
            return []
    
    def get_vision_embeddings(self, data):
        start_time = time.time()
        vision_hidden_states = self.chat_vpm(data)
        self.vision_time = time.time() - start_time
        vllm_embedding = F.embedding(data['input_ids'], self.embedding).to(self.device)
        new_vllm_embedding = vllm_embedding.clone()

        vision_hidden_states = [
            i.type(vllm_embedding.dtype) if isinstance(i, torch.Tensor) else i for i in vision_hidden_states
        ]

        '''fusion vision embedding with input embeds'''
        bs = len(data["input_ids"])
        for i in range(bs):
            cur_vs_hs = vision_hidden_states[i]
            if len(cur_vs_hs) > 0:
                cur_vllm_emb = vllm_embedding[i]
                cur_image_bound = data["image_bound"][i]
                if len(cur_image_bound) > 0:
                    image_indices = torch.stack(
                        [torch.arange(r[0], r[1], dtype=torch.long) for r in cur_image_bound]
                    ).to(vllm_embedding.device)

                    new_vllm_embedding[i] = cur_vllm_emb.scatter(
                        0,
                        image_indices.view(-1, 1).repeat(1, cur_vllm_emb.shape[-1]),
                        cur_vs_hs.view(-1, cur_vs_hs.shape[-1]),
                    ) 
        data["inputs_embeds"] = new_vllm_embedding
        return data
        #return new_vllm_embedding, vision_hidden_states

    def get_audio_embeddings(self, data):
        start_time = time.time()
        audio_embeddings = self.chat_audio(data)
        self.audio_time = time.time() - start_time
        input_embeddings = data["inputs_embeds"]

        '''fusion audio embeddings with input imbeds'''
        bs = len(input_embeddings)
        if len(data.get("audio_features", [])) > 0:
            assert len(audio_embeddings) == len(input_embeddings), "audio encoder output shape is different with audio input!"
            if len(audio_embeddings) > 0:
                audio_bounds = data["audio_bounds"]
                for i in range(bs):
                    if not audio_embeddings[i]:
                        continue
                    audio_embs = torch.cat(audio_embeddings[i], dim=0).to(
                        device=self.device, dtype=torch.float16
                    )
                    audio_start_pos = 0
                    for bound in audio_bounds[i]:
                        audio_len = bound[1] - bound[0]
                        input_embeddings[i, bound[0] : bound[1]] = audio_embs[
                                audio_start_pos : audio_start_pos + audio_len, :
                            ]
                        audio_start_pos += audio_len
        data["inputs_embeds"] = input_embeddings
        return data  

    def get_vap_out_embedding(self, data):
        data = self.get_vision_embeddings(data)

        data = self.get_audio_embeddings(data)

        return data
    
    def create_llm_prefill_inputs(self, inputs_embeds, pre_gen_idx):
        x = inputs_embeds[
            :,
            pre_gen_idx*self.llm_prefill_len : (pre_gen_idx + 1)*self.llm_prefill_len,
        ].detach().cpu().numpy()
        p_current_length = np.array(self.llm_prefill_len, dtype=self.llm_prefill_input_infos["current_length"].dtype)
        p_valid_length = np.array(p_current_length * pre_gen_idx, dtype=self.llm_prefill_input_infos["valid_length"].dtype)
        llm_prefill_inputs = dict(
            input_1=x,
            valid_length=p_valid_length,
            current_length=p_current_length
        )
        return llm_prefill_inputs
    
    def chat_llm_prefill(self, input_embeds):
        current_length = input_embeds.shape[1]
        if current_length >= self.llm_context_max_length:
            logger.error(f"Prefill input token length long than {self.llm_context_max_length}, please shorten it!")
            assert(0)
        last_hidden_states = None
        if current_length > self.llm_prefill_len:
            pre_gen_nums = current_length // self.llm_prefill_len
            for pre_gen_idx in range(pre_gen_nums):
                prefill_inputs = self.create_llm_prefill_inputs(input_embeds, pre_gen_idx)
                for i in range(3):
                    input_name = list(self.llm_prefill_input_infos.keys())[i]
                    self.llm_prefill_engine.set_input(input_name, prefill_inputs[input_name])
                self.llm_prefill_engine.run()
                self.llm_prefill_engine.sync()
                prefill_output = self.llm_prefill_engine.get_output(self.llm_output_names[0]).numpy()
                hidden_states = self.llm_prefill_engine.get_output(self.llm_output_names[1]).numpy()
                last_hidden_states = hidden_states if last_hidden_states is None else np.concatenate([last_hidden_states, hidden_states], axis=1)
        else:
            pre_gen_nums = 0
        current_length = current_length % self.llm_prefill_len
        prefill_shape = list(self.llm_prefill_shape)
        prefill_shape.append(self.hidden_dims)
        x = torch.zeros(prefill_shape, dtype=input_embeds.dtype, device=input_embeds.device)
        x[:, :current_length] = input_embeds[:, -current_length:]
        current_length = np.array(current_length, dtype=self.llm_prefill_input_infos["current_length"].dtype)
        valid_length = np.array(self.llm_prefill_len * pre_gen_nums, dtype=self.llm_decoder_input_infos["valid_length"].dtype) 
        inputs_list = [x.detach().cpu().numpy(), valid_length, current_length]
        for i in range(3):
            input_name = list(self.llm_prefill_input_infos.keys())[i]
            self.llm_prefill_engine.set_input(input_name, inputs_list[i])
        self.llm_prefill_engine.run()
        self.llm_prefill_engine.sync()
        prefill_output = self.llm_prefill_engine.get_output(self.llm_output_names[0]).numpy()
        hidden_states = self.llm_prefill_engine.get_output(self.llm_output_names[1]).numpy()[:, :current_length, ...]
        last_hidden_states = hidden_states if last_hidden_states is None else np.concatenate([last_hidden_states, hidden_states], axis=1)
        ### llm prefill post process   这一部分要修改
        next_token_logits = torch.from_numpy(prefill_output[:, -1, :]).to(dtype=torch.float32, device=self.device)
        if self.llm_sampling:
            next_token_scores = self.llm_prepared_logits_warper(torch.ones((1, 0), dtype=torch.long, device=self.device),
                                                                next_token_logits)
            # scores = torch.from_numpy(next_token_logits / self.llm_temperature).to(device=self.device, dtype=torch.float32)
            # top_k = min(self.llm_top_k, scores.size(-1))
            # indices_to_remove = scores < torch.topk(scores, top_k)[0][..., -1, None]
            # scores_processed = scores.masked_fill(indices_to_remove, self.llm_filter_value)
            # sorted_logits, sorted_indices = torch.sort(scores_processed, descending=False)
            # cumulative_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
            # sorted_indices_to_remove = cumulative_probs <= (1 - self.llm_top_p)
            # sorted_indices_to_remove[..., -self.llm_min_tokens_to_keep :] = 0
            # indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
            # next_token_scores = scores_processed.masked_fill(indices_to_remove, self.llm_filter_value)
            probs = F.softmax(next_token_scores, dim=-1)
            next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
        else:
            next_tokens = torch.argmax(next_token_logits, dim=-1)
        return next_tokens, last_hidden_states, valid_length, current_length
    
    def chat_llm_decoder(self, next_tokens:torch.Tensor, input_ids):
        input_embeds = F.embedding(next_tokens.unsqueeze(0), self.embedding)
        valid_length = np.array(self.context_length - 1, dtype=self.llm_decoder_input_infos["valid_length"].dtype)
        current_length = np.array(1, dtype=self.llm_decoder_input_infos["current_length"].dtype)
        inputs_list = [input_embeds.detach().cpu().numpy(), valid_length, current_length]
        for i in range(3):
            input_name = list(self.llm_decoder_input_infos.keys())[i]
            self.llm_decoder_engine.set_input(input_name, inputs_list[i])
        self.llm_decoder_engine.run()
        self.llm_decoder_engine.sync()
        decoder_output = self.llm_decoder_engine.get_output(self.llm_output_names[0]).numpy()
        hidden_states = self.llm_decoder_engine.get_output(self.llm_output_names[1]).numpy()
        next_token_logits = torch.from_numpy(decoder_output[:, -1, :]).to(self.device)
        # score = torch.gather(next_token_logits, 1, input_ids)
        # score = torch.where(score < 0, score * self.llm_penalty, score / self.llm_penalty)
        # scores_processed = next_token_logits.scatter(1, input_ids, score)
        if self.llm_sampling:
            next_token_scores = self.llm_prepared_logits_processer(input_ids, next_token_logits)
            next_token_scores = self.llm_prepared_logits_warper(input_ids, next_token_scores)
            # scores_processed = scores_processed / self.llm_temperature
            # top_k = min(self.llm_top_k, scores_processed.size(-1))
            # indices_to_remove = scores_processed < torch.topk(scores_processed, top_k[0][..., -1, None])
            # scores_processed = scores_processed.masked_fill(indices_to_remove, self.llm_filter_value)
            # sorted_logits, sorted_indices = torch.sort(scores_processed, descending=False)
            # cumulative_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
            # sorted_indices_to_remove = cumulative_probs <= (1 - self.llm_top_p)
            # sorted_indices_to_remove[..., -self.llm_min_tokens_to_keep :] = 0
            # indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
            # next_token_scores = scores_processed.masked_fill(indices_to_remove, self.llm_filter_value)
            probs = F.softmax(next_token_scores, dim=-1)
            next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
        else:
            next_tokens = torch.argmax(next_token_logits, dim=-1)
        return next_tokens, hidden_states

    def get_llm_out_tokens(self, data):
        input_ids = torch.ones((self.llm_prefill_shape[0], 0), dtype=torch.long, device=self.device)
        input_tokens_num = data["inputs_embeds"].shape[1]
        start_time = time.time()
        next_tokens, last_hidden_states, valid_length, current_length = self.chat_llm_prefill(data["inputs_embeds"])
        self.llm_prefill_time = time.time() - start_time
        self.llm_ttft_time = time.time() - self.llm_ttft_start_time
        input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
        self.context_length = valid_length + current_length + 1  # input tokens lengths
        output_token_num = 1
        this_peer_finished = False
        unfinished_sequences = torch.ones(input_ids.shape[0], dtype=torch.long, device=input_ids.device)
        #last_hidden_states = last_hidden_states[:, -1, ...]
        decode_start_time = time.time()
        while(~this_peer_finished):
            next_tokens, hidden_states = self.chat_llm_decoder(next_tokens, input_ids)
            last_hidden_states = np.concatenate([last_hidden_states, hidden_states], axis=1)
            self.context_length += 1
            input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
            if self.llm_sampling:
                unfinished_sequences = unfinished_sequences & ~self.llm_parpared_stopping_criteria(input_ids, None)
                this_peer_finished = unfinished_sequences.max() == 0
            else:
                this_peer_finished = (next_tokens == self.eos_token_id[0]).all()
            #this_peer_finished, unfinished_sequences = self.llm_criteria(input_ids, unfinished_sequences)
            output_token_num += 1
        self.llm_decode_time = time.time() - decode_start_time
        return input_ids, last_hidden_states, input_tokens_num, output_token_num
    
    def llm_decode_text(self, result_ids):
        result_text = []
        for result in result_ids:
            result = result[result != 0]
            if result[0] == self.tokenizer.bos_id:
                result = result[1:]
            if result[-1] in self.eos_token_id:
                result = result[:-1]
            result_text.append(self.tokenizer.decode(result))
        return result_text
    
    def merge_tts_inputs_embeds(self, input_ids, llm_spk_embeds):
        inputs_embeds = F.embedding(input_ids, self.tts_embedding).to(self.device)
        spk_emb_mask = input_ids == self.tts_spk_emb_token_id
        if spk_emb_mask.any():
            assert llm_spk_embeds is not None
            projector_input_name = list(self.tts_projector_input_infos.keys())[0]
            projector_input_data = llm_spk_embeds.detach().cpu().numpy().astype(self.tts_prefill_input_infos[projector_input_name].dtype)
            self.tts_projector_engine.set_input(projector_input_name, projector_input_data)
            self.tts_projector_engine.run()
            self.tts_projector_engine.sync()
            projected_spk_emb = self.tts_projector_engine.get_output(list(self.tts_projector_output_infos.keys())[0])
            projected_spk_emb = torch.from_numpy(projected_spk_emb).to(device=self.device, dtype=llm_spk_embeds.dtype)
            projected_spk_emb = F.normalize(projected_spk_emb, p=2, dim=-1)
            
            bs = input_ids.shape[0]
            for idx in range(bs):
                s_input_ids = input_ids[idx]
                s_spk_emb = projected_spk_emb[idx]
                s_mask = s_input_ids == self.tts_spk_emb_token_id
                nonzero_position_idx = s_mask.nonzero(as_tuple=False)
                assert nonzero_position_idx.shape[0] == self.tts_num_spk_embs
                begin_idx = nonzero_position_idx.min()
                end_idx = nonzero_position_idx.max()
                inputs_embeds[idx, begin_idx : end_idx+1, :] = s_spk_emb
        return inputs_embeds 

    def make_tts_streaming_chunk_mask(self, inputs_embeds, past_seen_tokens, streaming_tts_text_mask):
        assert inputs_embeds.shape[0] == 1
        dtype = inputs_embeds.dtype
        device = inputs_embeds.device
        min_dtype = torch.finfo(dtype).min

        causal_mask = torch.full((1, past_seen_tokens + inputs_embeds.shape[1]), full_value=0, dtype=dtype, device=device)

        invisible_text_tokens_start = (min(math.ceil((past_seen_tokens - self.tts_streaming_text_reserved_len) / self.tts_streaming_audio_chunk_size)
                                           * self.tts_streaming_text_chunk_size, self.tts_streaming_text_reserved_len,) + 1 + self.tts_num_spk_embs)
        invisible_text_tokens_end = (self.tts_streaming_text_reserved_len + 1 + self.tts_num_spk_embs + 1)

        causal_mask[0, invisible_text_tokens_start:invisible_text_tokens_end] = min_dtype

        causal_mask[0, 0 : 1 + self.tts_num_spk_embs + self.tts_streaming_text_reserved_len + 1].masked_fill_(streaming_tts_text_mask == 0, min_dtype)
        return causal_mask.unsqueeze(0).unsqueeze(0), min_dtype
    
    def chat_tts_prefill(self, input_embeds, current_length, valid_length, attention_mask):
        prefill_shape = list(self.tts_prefill_shape)
        x = torch.zeros(prefill_shape, dtype=input_embeds.dtype, device=input_embeds.device)
        x[:, :current_length] = input_embeds[:, -current_length:]
        current_length = np.array(current_length, dtype=self.tts_prefill_input_infos["current_length"].dtype)
        valid_length = np.array(valid_length, dtype=self.tts_prefill_input_infos["valid_length"].dtype) 
        inputs_list = [x.detach().cpu().numpy().astype(self.tts_prefill_input_infos["input_1"].dtype), 
                       current_length, 
                       valid_length, 
                       attention_mask.detach().cpu().numpy().astype(self.tts_prefill_input_infos["attention_mask"].dtype)]
        for i in range(4):
            input_name = list(self.tts_prefill_input_infos.keys())[i]
            self.tts_prefill_engine.set_input(input_name, inputs_list[i])
        self.tts_prefill_engine.run()
        self.tts_prefill_engine.sync()
    
    def chat_tts_decoder(self, input_embeds, valid_length, attention_mask):
        input_name = list(self.tts_decoder_input_infos.keys())[0]
        self.tts_decoder_engine.set_input(input_name,
                                          input_embeds.detach().cpu().numpy().astype(self.tts_decoder_input_infos[input_name].dtype))
        input_name = list(self.tts_decoder_input_infos.keys())[1]
        self.tts_decoder_engine.set_input(input_name,
                                          np.array(valid_length,dtype=self.tts_decoder_input_infos[input_name].dtype))
        input_name = list(self.tts_decoder_input_infos.keys())[3]
        self.tts_decoder_engine.set_input(input_name,
                                          attention_mask.detach().cpu().numpy().astype(self.tts_decoder_input_infos[input_name].dtype))
        self.tts_decoder_engine.run()
        self.tts_decoder_engine.sync()

        logits = self.tts_decoder_engine.get_output(list(self.tts_decoder_output_infos.keys())[0])
        logits = torch.from_numpy(logits).to(device=self.device, dtype=input_embeds.dtype)

        #### decoder post process
        logits = logits.permute(0, 2, 1)
        logits = logits.reshape(-1, logits.size(2))
        return logits

    def generate_tts_decoder(self, input_ids, streaming_tts_text_mask, valid_length):
        start_idx = 1 + self.tts_num_spk_embs * self.tts_use_speaker_embedding + self.tts_streaming_text_reserved_len + 1
        finish = torch.zeros(input_ids.shape[0], device=input_ids.device).bool()
        temperature = temperature.unsqueeze(0).expand(input_ids.shape[0], -1).contiguous().view(-1, 1)
        progress = input_ids.shape[1]
        input_ids_buf = torch.zeros(input_ids.shape[0], 
                                    progress + self.tts_max_new_token, 
                                    input_ids.shape[2], 
                                    dtype=input_ids.dtype, 
                                    device=input_ids.device)
        input_ids_buf.narrow(1, 0, progress).copy_(input_ids)
        del input_ids
        input_ids = input_ids_buf.narrow(1, 0, progress)
        condition_length = 1 + self.tts_num_spk_embs * self.tts_use_speaker_embedding + self.tts_streaming_text_reserved_len + 1

        for i in range(self.tts_max_new_token):
            audio_bos = True if progress == condition_length else False
            if audio_bos:
                narrowed_input_ids = torch.tensor([[self.tts_audio_bos_token_id]], dtype=torch.long, device=self.device)
                inputs_embeds = F.embedding(narrowed_input_ids, self.tts_embedding).to(self.device)
                del narrowed_input_ids
            else:
                narrowed_input_ids = input_ids.narrow(dim=1, start=input_ids.shape[1] - 1, length=1)
                code_emb = [F.embedding(self.tts_code_embeddings[i], narrowed_input_ids[:, :, i]) for i in range(self.tts_num_vq)]
                inputs_embeds = torch.stack(code_emb, 3).sum(3)
            
            causal_mask, min_dtype = self.make_tts_streaming_chunk_mask(inputs_embeds=inputs_embeds,
                                                             past_seen_tokens=valid_length,
                                                             streaming_tts_text_mask=streaming_tts_text_mask)
            attention_mask = torch.full(self.tts_mask_shape, fill_value=min_dtype, dtype=causal_mask.dtype, device=causal_mask.device)
            attention_mask[..., :causal_mask.shape[-1]] = causal_mask

            logits = self.chat_tts_decoder(inputs_embeds, valid_length, attention_mask)
            valid_length += 1

            #### tts decoder postprocess remain
            if self.tts_sampling:
                logits /= self.tts_temperature

                if not audio_bos:
                    input_ids_sliced = input_ids.narrow(1, start_idx, input_ids.size(1) - start_idx).permute(0, 2, 1)
                    logits_token = input_ids_sliced.reshape(input_ids_sliced.size(0) * input_ids_sliced.size(1), -1).to(self.device)
                    for logitsWarpers in self.tts_logits_warpers:
                        logits = logitsWarpers(logits_token, logits)
                
                if i < self.tts_min_new_token:
                    logits[:, self.tts_eos_token] = -torch.inf

                scores = F.softmax(logits, dim=-1)
                idx_next = torch.multinomial(scores, num_samples=1)
            else:
                idx_next = torch.argmax(logits, dim=-1)
            idx_next = idx_next.view(-1, self.tts_num_vq)
            finish_or = idx_next.eq(self.tts_eos_token).any(1)
            finish.logical_or_(finish_or)

            input_ids_buf.narrow(1, progress, 1).copy_(idx_next.unsqueeze_(1))
            if i == 0 and finish.any():
                break

            progress += 1
            input_ids = input_ids_buf.narrow(1, 0, progress)
            if finish.all():
                break
        if finish.all():
            genrated_input_ids = input_ids[:, condition_length:-1, :]
        else:
            genrated_input_ids = input_ids[:, condition_length:, :]
            
        return genrated_input_ids, input_ids, finish.all(), valid_length
    
    def generate_tts(self, inputs, last_hidden_states, text):
        last_hidden_states = torch.from_numpy(last_hidden_states).to(self.device)
        spk_bound = inputs["spk_bounds"][0][-1]
        spk_embeds = last_hidden_states[spk_bound[0] : spk_bound[1]]

        text = text.split("<|tts_bos|>")[-1]
        gen_text = text.split("<|tts_eos|>")[0]
        tts_tokens = self.tts_text_tokenizer.encode(gen_text, add_special_tokens=False)
        tts_tokens_len = len(tts_tokens)
        if tts_tokens_len < self.tts_streaming_text_reserved_len:
            num_pad_tokens = self.tts_streaming_text_reserved_len - tts_tokens_len
            pad_str = "[Etts]" + "[PAD]" * (num_pad_tokens - 1)
        else:
            tts_tokens = tts_tokens[0 : self.tts_streaming_text_reserved_len]
            tts_tokens_len = len(tts_tokens)
            text = self.tts_text_tokenizer.decode(tts_tokens, add_special_tokens=False)
            pad_str = ""
        spk_emb_placeholder_tts = "[spk_emb]" * self.tts_num_spk_embs
        new_text_tts = f"[Stts]{spk_emb_placeholder_tts}{text}{pad_str}[Ptts]"
        
        tts_inputs = self.tts_text_tokenizer.encode(new_text_tts, add_special_tokens=False)
        tts_input_ids = torch.Tensor(tts_inputs).unsqueeze(0).to(self.device, dtype=torch.long)
        
        tts_squence_full_length = (
            1 + self.tts_num_spk_embs * self.tts_use_speaker_embedding + self.tts_streaming_text_reserved_len + 1
        )
        streaming_attention_mask = torch.zeros(tts_squence_full_length, dtype=torch.int8)
        streaming_attention_mask[0 : 1 + 1 + tts_tokens_len + 1] = 1
        streaming_attention_mask[-1] = 1

        streaming_tts_text_mask = streaming_attention_mask.to(self.device)
        
        condition_length = (
            1 + self.tts_use_speaker_embedding * self.tts_num_spk_embs + self.tts_streaming_text_reserved_len + 1
        )
        emb = torch.zeros(1, condition_length, self.tts_num_vq, dtype=torch.float16, device=self.device)

        audio_input_ids = torch.zeros(1, condition_length, self.tts_num_vq, dtype=torch.long, device=self.device)
        
        eos_lab = False
        tts_prefill_valid_length = 0
        tts_decoder_valid_length = condition_length - 1
        for chunk_idx in range(math.ceil(emb.shape[1]) / self.tts_streaming_text_chunk_size):
            if chunk_idx == 0:
                begin = chunk_idx * self.tts_streaming_text_chunk_size + 0
                end = (chunk_idx + 1) * self.tts_streaming_text_chunk_size + 1 + self.tts_use_speaker_embedding * self.tts_num_spk_embs
            else:
                begin = chunk_idx * self.tts_streaming_text_chunk_size + 1 + self.tts_use_speaker_embedding * self.tts_num_spk_embs
                end = min((chunk_idx + 1) * self.tts_streaming_text_chunk_size + 1 + self.tts_use_speaker_embedding * self.tts_num_spk_embs, condition_length - 1)
            if end - begin > 0:
                text_input_ids = tts_input_ids[:, begin:end]
                tts_prefill_input_embeds = self.merge_tts_inputs_embeds(text_input_ids, spk_embeds if begin == 0 else None)
                min_dtype = torch.finfo(tts_prefill_input_embeds.dtype).min
                tts_prefill_current_length = tts_prefill_input_embeds.shape[1]
                tts_prefill_attention_mask = torch.zeros(self.tts_mask_shape, dtype=self.tts_mask_dtype, device=self.device)
                tts_prefill_attention_mask[..., tts_prefill_current_length:] = min_dtype
                self.chat_tts_prefill(tts_prefill_input_embeds,
                                      tts_prefill_current_length,
                                      tts_prefill_valid_length,
                                      tts_prefill_attention_mask)
                tts_prefill_valid_length += tts_prefill_current_length
            
            audio_input_ids, finished, new_ids, tts_decoder_valid_length = self.generate_tts_decoder(audio_input_ids, streaming_tts_text_mask, tts_decoder_valid_length)

            if finished:
                logger.info("tts generation finished!")
                eos_lab = True
                break
        if not eos_lab:
            logger.warning("tts eos_lab False, Generation continue!")
            while True:
                audio_input_ids, finished, new_ids, tts_decoder_valid_length = self.generate_tts_decoder(audio_input_ids, 
                                                                                                         streaming_tts_text_mask, 
                                                                                                         tts_decoder_valid_length)
                if finished:
                    logger.info("Not eos_lab Generation finished.")
                    break
                if new_ids.shape[1] > self.tts_context_max_length:
                    logger.warning(f"tts Generation length > {self.tts_context_max_length}, stopped!")
                    break
        return new_ids
            
    def dvae_decode_to_mel_spec(self, result_list: list[torch.Tensor]):
        max_x_len = -1
        if len(result_list) == 0:
            return np.array([], dtype=np.float32)
        for result in result_list:
            if result.size(0) > max_x_len:
                max_x_len = result.size(0)
        batch_result = torch.zeros(
            (len(result_list), result_list[0].size(1), max_x_len),
            dtype=result_list[0].dtype,
            device=result_list[0].device
        )  
        for i in range(len(result_list)):
            src = result_list[i]
            batch_result[i].narrow(1, 0, src.size(0)).copy_(src.permute(1, 0))
        
        input_name = list(self.dvae_input_infos.keys())[0]
        input_shape = self.dvae_input_infos[input_name].shape
        input_dtype = self.dvae_input_infos[input_name].dtype

        truth_input = np.zeros(input_shape, dtype=input_dtype)
        if input_shape[-1] >= batch_result.shape[-1]:
            truth_input[..., :batch_result.shape[-1]] = batch_result.detach().cpu().numpy()
        else:
            truth_input = batch_result.detach().cpu().numpy()[..., :input_shape[-1]].astype(input_dtype)
        
        self.dvae_engine.set_input(input_name, truth_input)
        self.dvae_engine.run()
        self.dvae_engine.sync()

        mel_specs = self.dvae_engine.get_output(list(self.dvae_output_infos.keys())[0])
        mel_specs = np.reshape(mel_specs, (mel_specs.shape[0], mel_specs.shape[1], -1, 2))
        if input_shape[-1] > batch_result.shape[-1]:
            mel_specs = mel_specs[:, :, :input_shape[-1]]
        mel_specs = np.reshape(mel_specs, (mel_specs.shape[0], mel_specs.shape[1], -1))

        return mel_specs

    def vocos_decode_mel(self, mel_spec: np.ndarray):
        input_name = list(self.vocos_input_infos.keys())[0]
        input_shape = self.vocos_input_infos[input_name].shape
        input_dtype = self.vocos_input_infos[input_name].dtype

        mel_spec = mel_spec.astype(input_dtype)

        truth_input = np.zeros(input_shape, dtype=input_dtype)
        if input_shape[-1] >= mel_spec.shape[-1]:
            truth_input[..., :mel_spec.shape[-1]] = mel_spec
            truth_length = mel_spec.shape[-1]
        else:
            truth_input = mel_spec[..., :input_shape[-1]]
            truth_length = input_shape[-1]
        
        self.vocos_engine.set_input(input_name, truth_input)
        self.vocos_engine.run()
        self.vocos_engine.sync()

        #### 这部分onnx,少一个slice mag的输出，但其实exp和clip应该也可以放到模型中
        #output_names = sorted(list(self.vocos_output_infos.keys()))
        mag = self.vocos_engine.get_output("slice_output_0")[..., :truth_length]
        x = self.vocos_engine.get_output("x")[..., :truth_length]
        y = self.vocos_engine.get_output("y")[..., :truth_length]

        mag = torch.from_numpy(mag).to(device=self.device, dtype=torch.float16)
        mag = torch.exp(mag)
        mag = torch.clip(mag, max=1e2)

        S = mag * (x + 1j * y)
        audio = self.vocos_istft(S)
        return audio

    def generate(
            self,
            input_ids=None,
            pixel_values=None,
            tgt_sizes=None,
            audio_features=[],
            audio_feature_lens=None,
            image_bound=None,
            audio_bounds=None,
            spk_bounds=None,
            vision_hidden_states=None,
            **kwargs,
        ):
        
        assert input_ids is not None
        assert len(input_ids) == len(pixel_values)

        model_inputs = {
            "input_ids": input_ids,
            "audio_features": audio_features,
            "audio_feature_lens": audio_feature_lens,
            "image_bound": image_bound,
            "audio_bounds": audio_bounds,
            "spk_bounds": spk_bounds,
        }

        if vision_hidden_states is None:
            model_inputs["pixel_values"] = pixel_values
            model_inputs["tgt_sizes"] = tgt_sizes
        else:
            model_inputs["vision_hidden_states"] = vision_hidden_states
        self.llm_ttft_start_time = time.time()
        model_inputs = self.get_vap_out_embedding(model_inputs)
        token_ids, last_hidden_states, input_tokens_num, output_token_nums = self.get_llm_out_tokens(model_inputs)
        result = self.llm_decode_text(token_ids)
        answer = result[0]

        generate_audio = kwargs.get("generate_audio", False)
        if self.use_tts_template and generate_audio:
            if not self.init_tts:
                logger.warning("TTS model not init, can not generate audio! Will only return text answer.")
                pass
            output_audio_path = kwargs.get("output_audio_path", "./output.wav")
            ### get mel spectrum(梅尔频谱)
            tts_tokens_ids = self.generate_tts(model_inputs, last_hidden_states, answer)
            mel_spec = self.dvae_decode_to_mel_spec(tts_tokens_ids)
            wav_numpy = self.vocos_decode_mel(mel_spec).detach().cpu().numpy()
            if 1:
                sf.write(output_audio_path, wav_numpy, samplerate=self.wav_sr)
                logger.info(f"Audio saved to {output_audio_path}")
        return answer, input_tokens_num, output_token_nums

    def chat(self, msgs, **kwargs):
        if isinstance(msgs[0], list):
            batched = True
        else:
            batched = False

        msgs_list = msgs

        if batched is False:
            msgs_list = [msgs_list]
        max_slice_nums = kwargs.get("max_slice_nums", None)
        omni_input = kwargs.get("omni_input", False)
        
        inputs = self.regularize_msgs(msgs_list, max_slice_nums, omni_input)

        answer, input_tokens_num, output_token_nums = self.generate(input_ids=inputs["input_ids"],
                               pixel_values=inputs["pixel_values"],
                               tgt_sizes=inputs["tgt_sizes"],
                               audio_features=inputs["audio_features"],
                               audio_feature_lens=inputs["audio_feature_lens"],
                               image_bound=inputs["image_bound"],
                               audio_bounds=inputs["audio_bounds"],
                               spk_bounds=inputs["spk_bounds"],
                               vision_hidden_states=None,
                               **kwargs)
        answer = answer.replace(self.tokenizer.tts_end, "")

        return answer, input_tokens_num, output_token_nums
    
def xh2_demo(args):
    hmminicpmo = HMMiniCPMO(args,
                init_vision=True,
                init_audio=False,
                init_tts=False,
                use_tts_template=False,
                llm_sampling=False,
                tts_sampling=False)

    example_mode = EXAMPLES_MODE[args.example_idx]
    if example_mode == "all":
        video_path="MiniCPM-o-2_6/assets/Skiing.mp4"
        # if use voice clone prompt, please set ref_audio
        ref_audio_path = 'MiniCPM-o-2_6/assets/demo.wav'
        ref_audio, _ = librosa.load(ref_audio_path, sr=16000, mono=True)
        sys_msg = {"role": "user",
                "content":["你是一个AI助手。你能接受视频，音频和文本输入并输出语音和文本。模仿输入音频中的声音特征。",
                            ref_audio,
                            "作为助手，你将使用这种声音风格说话。"]}
        video_contents = get_video_chunk_content(video_path, flatten=False)
        msg = {"role":"user", "content": video_contents}
        #msg = {"role":"user", "content": contents}
        msgs = [sys_msg, msg]
        logger.info(msgs)
        start_time = time.time()
        answer, input_tokens_num, output_tokens_num = hmminicpmo.chat(
            msgs=msgs,
            generate_audio=True,
            max_slice_nums=1,
            omni_input=True,
            output_audio_path="./output_hmm.wav",
        )
        total_time = time.time() - start_time
        logger.info(f"{answer}")
        logger.success(f"Total cost {total_time * 1000:.3f} ms")
        logger.success(f"Input tokens: {input_tokens_num}, Output tokens: {output_tokens_num}")
        logger.success(f"Vision Cost {hmminicpmo.vision_time * 1000:.3f} ms")
        logger.success(f"LLM Prefill Speed: {input_tokens_num / hmminicpmo.llm_prefill_time:.2f} tokens/s")
        logger.success(f"TTFT (Time to First Token): {hmminicpmo.llm_ttft_time * 1000:.3f} ms")
        logger.success(f"TPOT (Time Per Output Token): {(output_tokens_num - 1) / hmminicpmo.llm_decode_time:.2f} tokens/s")
        logger.success(f"E2E Latency (End-to-End Latency): {total_time:.3f} seconds")
        logger.success(f"TPS (Tokens Per Second): {output_tokens_num / total_time:.2f} tokens/s")

    elif example_mode == "llm":
        msg = {"role":"user", "content": ["介绍一下存算一体技术。"]}
        msgs = [msg]
        logger.info(msgs)
        start_time = time.time()
        answer, input_tokens_num, output_tokens_num = hmminicpmo.chat(
            msgs=msgs
        )
        total_time = time.time() - start_time
        logger.info(f"{answer}")
        logger.success(f"Total cost {total_time * 1000:.3f} ms")
        logger.success(f"Input tokens: {input_tokens_num}, Output tokens: {output_tokens_num}")
        logger.success(f"LLM Prefill Speed: {input_tokens_num / hmminicpmo.llm_prefill_time:.2f} tokens/s")
        logger.success(f"TTFT (Time to First Token): {hmminicpmo.llm_ttft_time * 1000:.3f} ms")
        logger.success(f"TPOT (Time Per Output Token): {(output_tokens_num - 1) / hmminicpmo.llm_decode_time:.2f} tokens/s")
        logger.success(f"E2E Latency (End-to-End Latency): {total_time:.3f} seconds")
        logger.success(f"TPS (Tokens Per Second): {output_tokens_num / total_time:.2f} tokens/s")

    elif example_mode == "vllm":
        image = Image.open("MiniCPM-o-2_6/airplane.jpeg").convert("RGB")
        msg = {"role":"user", "content": [image, "图中是哪家航空公司的飞机？"]}
        msgs = [msg]
        logger.info(msgs)
        start_time = time.time()
        answer, input_tokens_num, output_tokens_num = hmminicpmo.chat(
            msgs=msgs
        )
        total_time = time.time() - start_time
        logger.info(f"{answer}")
        logger.success(f"Total cost {total_time * 1000:.3f} ms")
        logger.success(f"Input tokens: {input_tokens_num}, Output tokens: {output_tokens_num}")
        logger.success(f"Vision Cost {hmminicpmo.vision_time * 1000:.3f} ms")
        logger.success(f"LLM Prefill Speed: {input_tokens_num / hmminicpmo.llm_prefill_time:.2f} tokens/s")
        logger.success(f"TTFT (Time to First Token): {hmminicpmo.llm_ttft_time * 1000:.3f} ms")
        logger.success(f"TPOT (Time Per Output Token): {(output_tokens_num - 1) / hmminicpmo.llm_decode_time:.2f} tokens/s")
        logger.success(f"E2E Latency (End-to-End Latency): {total_time:.3f} seconds")
        logger.success(f"TPS (Tokens Per Second): {output_tokens_num / total_time:.2f} tokens/s")

    elif example_mode == "mvllm":
        image0 = Image.open("./MiniCPM-o-2_6/000000002532.jpg").convert("RGB")
        image1 = Image.open("./MiniCPM-o-2_6/000000001268.jpg").convert("RGB")
        msg = {"role":"user", "content": [image0, image1, "两张图的不同之处？"]}
        msgs = [msg]
        logger.info(msgs)
        start_time = time.time()
        answer, input_tokens_num, output_tokens_num = hmminicpmo.chat(
            msgs=msgs
        )
        total_time = time.time() - start_time
        logger.info(f"{answer}")
        logger.success(f"Total cost {total_time * 1000:.3f} ms")
        logger.success(f"Input tokens: {input_tokens_num}, Output tokens: {output_tokens_num}")
        logger.success(f"Vision Cost {hmminicpmo.vision_time * 1000:.3f} ms")
        logger.success(f"LLM Prefill Speed: {input_tokens_num / hmminicpmo.llm_prefill_time:.2f} tokens/s")
        logger.success(f"TTFT (Time to First Token): {hmminicpmo.llm_ttft_time * 1000:.3f} ms")
        logger.success(f"TPOT (Time Per Output Token): {(output_tokens_num - 1) / hmminicpmo.llm_decode_time:.2f} tokens/s")
        logger.success(f"E2E Latency (End-to-End Latency): {total_time:.3f} seconds")
        logger.success(f"TPS (Tokens Per Second): {output_tokens_num / total_time:.2f} tokens/s")

    else:
        logger.error(f"not support example mode {example_mode}!")
        assert(0)

if __name__ == "__main__":

    args = get_args()
    if HOUMO_TARGET == 'xh1':
        logger.error(f"bge embedder or reranker is not support xh1 platform!")
        assert(0)
    elif HOUMO_TARGET == 'xh2':
        xh2_demo(args)
