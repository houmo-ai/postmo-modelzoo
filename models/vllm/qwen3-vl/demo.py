import os
import re
import sys
import time
import logging
import argparse
from typing import List, Optional, Tuple, Union
from loguru import logger

logging.basicConfig(level=logging.ERROR)
import warnings

warnings.simplefilter(action="ignore", category=UserWarning)
warnings.simplefilter(action="ignore", category=FutureWarning)
import torch
import torch.nn.functional as F
import numpy as np
import tcim_lite as tcim
from PIL import Image
from processing_qwen3_vl import Qwen3VLProcessor
from utils import get_rope_index, QRawToYuv

HOUMO_TARGET = os.getenv("HOUMO_TARGET")
assert HOUMO_TARGET in ["xh2"], f"Unsupported HOUMO_TARGET: {HOUMO_TARGET}"


def is_valid_char(cp):
    if (
        (cp >= 0x4E00 and cp <= 0x9FFF)
        or (cp >= 0x3400 and cp <= 0x4DBF)
        or (cp >= 0x20000 and cp <= 0x2A6DF)
        or (cp >= 0x2A700 and cp <= 0x2B73F)
        or (cp >= 0x2B740 and cp <= 0x2B81F)
        or (cp >= 0x2B820 and cp <= 0x2CEAF)
        or (cp >= 0xF900 and cp <= 0xFAFF)
        or (cp >= 0x2F800 and cp <= 0x2FA1F)
        or (0x0041 <= cp and cp <= 0x005A)
        or (0x0061 <= cp and cp <= 0x007A)
    ):
        return True

    return False


def get_args() -> argparse.Namespace:
    """Parse commandline."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tokenizer_dir",
        dest="tokenizer_dir",
        type=str,
        default="qwen3-vl",
        help="tokenizer dir",
    )
    parser.add_argument(
        "--embedding_path",
        dest="embedding_path",
        type=str,
        default=os.path.join("output", HOUMO_TARGET, "hmquant", "quant_embedding.pt"),
        help="houmo embedding weight path",
    )
    parser.add_argument(
        "--vit_path",
        dest="vit_path",
        type=str,
        default=os.path.join("output", HOUMO_TARGET, "qwen3-vl_visual.hmm"),
        help="houmo visual model path",
    )
    parser.add_argument(
        "--prefill_path",
        dest="prefill_path",
        type=str,
        default=os.path.join("output", HOUMO_TARGET, "qwen3-vl_prefill.hmm"),
        help="houmo prefill model path",
    )
    parser.add_argument(
        "--decode_path",
        dest="decode_path",
        type=str,
        default=os.path.join("output", HOUMO_TARGET, "qwen3-vl_decode.hmm"),
        help="houmo decode model path",
    )
    parser.add_argument(
        "--repetition_penalty",
        dest="repetition_penalty",
        type=float,
        default=1.0,
        help="sampling repetition_penalty",
    )
    parser.add_argument(
        "--topk",
        dest="topk",
        type=int,
        default=None,
        help="sampling top-k",
    )
    parser.add_argument(
        "--topp",
        dest="topp",
        type=float,
        default=1.0,
        help="sampling top-p",
    )
    parser.add_argument(
        "--temperature",
        dest="temperature",
        type=float,
        default=1.0,
        help="sampling temperature",
    )
    args = parser.parse_args()
    return args


class SamplingManager:
    def __init__(
        self,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: float = 1.0,
        repetition_penalty: float = 1.0,
        min_tokens_to_keep: int = 1,
    ):
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.repetition_penalty = repetition_penalty
        self.min_tokens_to_keep = min_tokens_to_keep

    def softmax(self, x: np.ndarray) -> np.ndarray:
        exp_x = np.exp(x - np.max(x))
        return exp_x / np.sum(exp_x)

    def apply_temperature(self, logits: np.ndarray) -> np.ndarray:
        if self.temperature <= 0:
            raise ValueError("Temperature must larger than 0")

        return logits / self.temperature

    def apply_repetition_penalty(
        self, logits: np.ndarray, previous_tokens: Optional[List[int]] = None
    ) -> np.ndarray:
        if self.repetition_penalty == 1.0 or not previous_tokens:
            return logits

        adjusted_logits = logits.copy()
        for token_id in set(previous_tokens):
            if 0 <= token_id < len(logits):
                if logits[token_id] < 0:
                    adjusted_logits[token_id] = (
                        logits[token_id] * self.repetition_penalty
                    )
                else:
                    adjusted_logits[token_id] = (
                        logits[token_id] / self.repetition_penalty
                    )

        return adjusted_logits

    def apply_top_k(self, probs: np.ndarray) -> np.ndarray:
        if self.top_k is None or self.top_k <= 0:
            return probs

        top_k = min(self.top_k, len(probs))

        if top_k <= 0:
            return probs

        top_k_indices = np.argpartition(probs, -top_k)[-top_k:]

        mask = np.ones_like(probs, dtype=bool)
        mask[top_k_indices] = False
        filtered_probs = probs.copy()
        filtered_probs[mask] = 0

        if np.sum(filtered_probs) > 0:
            normalized_probs = filtered_probs / np.sum(filtered_probs)
        else:
            normalized_probs = np.ones_like(probs) / len(probs)

        return normalized_probs

    def apply_top_p(self, probs: np.ndarray) -> np.ndarray:
        if self.top_p >= 1.0:
            return probs

        sorted_indices = np.argsort(probs)[::-1]
        sorted_probs = probs[sorted_indices]

        cumulative_probs = np.cumsum(sorted_probs)

        cutoff_indices = np.where(cumulative_probs >= self.top_p)[0]

        if len(cutoff_indices) > 0:
            cutoff_index = cutoff_indices[0]
            if cutoff_index < self.min_tokens_to_keep - 1:
                cutoff_index = self.min_tokens_to_keep - 1

            selected_indices = sorted_indices[: cutoff_index + 1]
        else:
            selected_indices = sorted_indices

        mask = np.ones_like(probs, dtype=bool)
        mask[selected_indices] = False
        filtered_probs = probs.copy()
        filtered_probs[mask] = 0

        if np.sum(filtered_probs) > 0:
            normalized_probs = filtered_probs / np.sum(filtered_probs)
        else:
            normalized_probs = np.ones_like(probs) / len(probs)

        return normalized_probs

    def process_logits(
        self, logits: np.ndarray, previous_tokens: Optional[List[int]] = None
    ) -> np.ndarray:
        processed_logits = logits.copy()
        # 1. apply repetition penalty
        processed_logits = self.apply_repetition_penalty(
            processed_logits, previous_tokens
        )

        # 2. apply softmax
        # not using softmax in case of long time cost
        probs = processed_logits
        # probs = self.softmax(processed_logits)

        # 3. apply top-k
        probs = self.apply_top_k(probs)

        # 4. apply top-p
        probs = self.apply_top_p(probs)

        # 5. apply temperature
        probs = self.apply_temperature(probs)
        return probs

    def sample(
        self, logits: np.ndarray, previous_tokens: Optional[List[int]] = None
    ) -> int:
        logits = logits[0][0]
        probs = self.process_logits(logits, previous_tokens)
        if np.all(probs == 0):
            probs = np.ones_like(probs) / len(probs)

        # sampled_index = np.random.choice(len(probs), p=probs)
        sampled_index = probs.argmax(-1)

        return np.array([[sampled_index]])

    def get_processed_probs(
        self, logits: np.ndarray, previous_tokens: Optional[List[int]] = None
    ) -> np.ndarray:
        return self.process_logits(logits, previous_tokens)


class Qwen3VL:
    def __init__(
        self,
        vit_path,
        prefill_path,
        decode_path,
        tokenizer_dir,
        embedding_path,
    ):
        weight_manager = tcim.runtime.WeightManager(0)
        option0 = tcim.runtime.Option(weight_manager)
        option1 = tcim.runtime.Option(weight_manager)
        option2 = tcim.runtime.Option(weight_manager)
        self.vit_model = tcim.runtime.load(os.path.join(vit_path), option=option0)
        logger.info("vit model loaded")
        self.prefill = tcim.runtime.load(os.path.join(prefill_path), option=option1)
        logger.info("prefill model loaded")
        self.input_names = self.get_input_names()
        dummy_tensor_names = []
        for input_name in self.input_names:
            if "model_layers" in input_name:
                dummy_tensor_names.append(input_name)
        option1.set_dummy_tensors(dummy_tensor_names)
        self.decode = tcim.runtime.load(os.path.join(decode_path), option=option2)
        logger.info("decode model loaded")
        self.samplingmanager = SamplingManager(
            temperature=args.temperature,
            top_k=args.topk,
            top_p=args.topp,
            repetition_penalty=args.repetition_penalty,
        )
        self.processor = Qwen3VLProcessor.from_pretrained(tokenizer_dir)
        self.device = torch.device("cpu")
        prefill_shape = self.prefill.get_input_info(
            self.prefill.get_input_name(0)
        ).shape[:2]
        self.prefill_shape = torch.Size(prefill_shape)
        self.prefill_len = self.prefill_shape.numel()
        self.pad_token_id = 0
        self.image_token_id = 151655
        self.video_token_id = 151656
        self.vision_start_token_id = 151652
        self.vision_end_token_id = 151653
        self.vision_token_id = 151654
        self.eos_token_id = [151645, 151643]
        self.spatial_merge_size = 2
        self.patch_size = 16
        self.max_size_t = 2
        self.temporal_patch_size = 2
        self.spatial_merge_unit = self.spatial_merge_size * self.spatial_merge_size
        self.batch_size = 1
        self.resize_v1 = True
        # set mode
        self.rgb2yuv = QRawToYuv(input_color_type="RGB", toYUV_format="YUV444")

        self.embedding_len = self.prefill.get_input_info(
            self.prefill.get_input_name(0)
        ).shape[2]
        self.context_max_length = self.decode.get_input_info(
            self.decode.get_input_name(10)
        ).shape[2]
        self.image_shape = self.vit_model.get_input_info(
            self.vit_model.get_input_name(0)
        ).shape[-2:]
        self.image_size_w = self.image_shape[0]
        self.image_size_h = self.image_shape[1]
        for input_name in self.input_names:
            if "model_layers" in input_name:
                cache = self.decode.get_dev_input(input_name)
                self.prefill.set_input(input_name, cache)
        self.decode.set_input("current_length", np.array([1]).astype("int16"))
        self.embedding = torch.load(embedding_path, weights_only=False)
        if HOUMO_TARGET == "xh2":
            self.embedding = self.embedding.weight
        self.hidden_dims = self.embedding.shape[-1]

    def get_input_names(self):
        input_names = []
        for i in range(self.prefill.get_num_inputs()):
            input_names.append(self.prefill.get_input_name(i))
        return input_names

    def get_input_name(self, model):
        input_names = []
        for i in range(model.get_num_inputs()):
            input_names.append(model.get_input_name(i))
        return input_names

    def get_output_name(self, model):
        output_names = []
        for i in range(model.get_num_outputs()):
            output_names.append(model.get_output_name(i))
        return output_names

    def create_template(self, prompt, image_dir=None):
        if image_dir is None:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
        else:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": image_dir,
                        },
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
        return messages

    def preprocess(self, prompt, image_dir, processor):
        from qwen_vl_utils import process_vision_info

        messages = self.create_template(prompt, image_dir)
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        if image_dir is not None:
            image_inputs, video_inputs = process_vision_info(
                messages, image_patch_size=self.patch_size
            )
        else:
            image_inputs, video_inputs = None, None
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        return inputs

    def preprocess_visual(self, inputs):
        visual_inputs = dict()
        visual_inputs["hidden_states"] = inputs["hm_pixel_values"][0]
        visual_inputs["hidden_states"] = visual_inputs["hidden_states"].to(self.device)
        return (visual_inputs["hidden_states"].half(),)

    def load_and_process_image(self, image_path):
        """
        Loads an image from the given path, converts to RGB, resizes proportionally if needed,
        and pads to (self.image_size_w, self.image_size_h) with (114,114,114) background.
        Returns the processed PIL image.
        """
        from PIL import Image, ImageOps

        target_w, target_h = self.image_size_w, self.image_size_h
        image = Image.open(image_path).convert("RGB")
        orig_w, orig_h = image.size
        if (orig_w, orig_h) != (target_w, target_h):
            # Resize while keeping aspect ratio
            scale = min(target_w / orig_w, target_h / orig_h)
            new_w = int(orig_w * scale)
            new_h = int(orig_h * scale)
            image = image.resize((new_w, new_h), Image.BICUBIC)
            # Pad to target size
            pad_w = target_w - new_w
            pad_h = target_h - new_h
            left = 0
            top = 0
            right = pad_w
            bottom = pad_h
            image = ImageOps.expand(
                image, border=(left, top, right, bottom), fill=(114, 114, 114)
            )
        return image

    def load_and_process_image_v2(self, image_path):
        """
        Loads an image from the given path, converts to RGB, resizes proportionally if needed,
        and pads to (self.image_size_w, self.image_size_h) with (114,114,114) background.
        Returns the processed PIL image.
        """
        from PIL import Image, ImageOps

        target_w, target_h = self.image_size_w, self.image_size_h
        image = Image.open(image_path).convert("RGB")
        orig_w, orig_h = image.size
        if (orig_w, orig_h) != (target_w, target_h):
            image = image.resize((target_w, target_h), Image.BICUBIC)
        return image

    def preprocess_prefill(self, data):
        input_ids = data["input_ids"]
        input_seq_len = input_ids.shape[-1]

        steps = (
            input_seq_len + self.prefill_input_sequence_length - 1
        ) // self.prefill_input_sequence_length

        self.input_sequence_length = self.prefill_input_sequence_length * steps
        inputs = self.prepare_inputs(data)

        return inputs

    def get_rope_index(
        self,
        input_ids: torch.LongTensor,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calculate the 3D rope index based on image and video's temporal, height and width in LLM.

        Explanation:
            Each embedding sequence contains vision embedding and text embedding or just contains text embedding.

            For pure text embedding sequence, the rotary position embedding has no difference with mordern LLMs.
            Examples:
                input_ids: [T T T T T], here T is for text.
                temporal position_ids: [0, 1, 2, 3, 4]
                height position_ids: [0, 1, 2, 3, 4]
                width position_ids: [0, 1, 2, 3, 4]

            For vision and text embedding sequence, we calculate 3D rotary position embedding for vision part
            and 1D rotary position embeddin for text part.
            Examples:
                Assume we have a video input with 3 temporal patches, 2 height patches and 2 width patches.
                input_ids: [V V V V V V V V V V V V T T T T T], here V is for vision.
                vision temporal position_ids: [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2]
                vision height position_ids: [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1]
                vision width position_ids: [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
                text temporal position_ids: [3, 4, 5, 6, 7]
                text height position_ids: [3, 4, 5, 6, 7]
                text width position_ids: [3, 4, 5, 6, 7]
                Here we calculate the text start position_ids as the max vision position_ids plus 1.

        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
                it.
            image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
                The temporal, height and width of feature shape of each image in LLM.
            video_grid_thw (`torch.LongTensor` of shape `(num_videos, 3)`, *optional*):
                The temporal, height and width of feature shape of each video in LLM.
            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

        Returns:
            position_ids (`torch.LongTensor` of shape `(3, batch_size, sequence_length)`)
            mrope_position_deltas (`torch.Tensor` of shape `(batch_size)`)
        """
        spatial_merge_size = self.spatial_merge_size
        image_token_id = self.image_token_id
        video_token_id = self.video_token_id
        vision_start_token_id = self.vision_start_token_id
        mrope_position_deltas = []
        if input_ids is not None and (
            image_grid_thw is not None or video_grid_thw is not None
        ):
            total_input_ids = input_ids
            if attention_mask is None:
                attention_mask = torch.ones_like(total_input_ids)
            position_ids = torch.ones(
                3,
                input_ids.shape[0],
                input_ids.shape[1],
                dtype=input_ids.dtype,
                device=input_ids.device,
            )
            image_index, video_index = 0, 0
            for i, input_ids in enumerate(total_input_ids):
                input_ids = input_ids[attention_mask[i] == 1]
                image_nums, video_nums = 0, 0
                vision_start_indices = torch.argwhere(
                    input_ids == vision_start_token_id
                ).squeeze(1)
                vision_tokens = input_ids[vision_start_indices + 1]
                image_nums = (vision_tokens == image_token_id).sum()
                video_nums = (vision_tokens == video_token_id).sum()
                input_tokens = input_ids.tolist()
                llm_pos_ids_list: list = []
                st = 0
                remain_images, remain_videos = image_nums, video_nums
                for _ in range(image_nums + video_nums):
                    if image_token_id in input_tokens and remain_images > 0:
                        ed_image = input_tokens.index(image_token_id, st)
                    else:
                        ed_image = len(input_tokens) + 1
                    if video_token_id in input_tokens and remain_videos > 0:
                        ed_video = input_tokens.index(video_token_id, st)
                    else:
                        ed_video = len(input_tokens) + 1
                    if ed_image < ed_video:
                        t, h, w = (
                            image_grid_thw[image_index][0],
                            image_grid_thw[image_index][1],
                            image_grid_thw[image_index][2],
                        )
                        image_index += 1
                        remain_images -= 1
                        ed = ed_image
                    else:
                        t, h, w = (
                            video_grid_thw[video_index][0],
                            video_grid_thw[video_index][1],
                            video_grid_thw[video_index][2],
                        )
                        video_index += 1
                        remain_videos -= 1
                        ed = ed_video
                    llm_grid_t, llm_grid_h, llm_grid_w = (
                        t.item(),
                        h.item() // spatial_merge_size,
                        w.item() // spatial_merge_size,
                    )
                    text_len = ed - st

                    st_idx = (
                        llm_pos_ids_list[-1].max() + 1
                        if len(llm_pos_ids_list) > 0
                        else 0
                    )
                    llm_pos_ids_list.append(
                        torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx
                    )

                    t_index = (
                        torch.arange(llm_grid_t)
                        .view(-1, 1)
                        .expand(-1, llm_grid_h * llm_grid_w)
                        .flatten()
                    )
                    h_index = (
                        torch.arange(llm_grid_h)
                        .view(1, -1, 1)
                        .expand(llm_grid_t, -1, llm_grid_w)
                        .flatten()
                    )
                    w_index = (
                        torch.arange(llm_grid_w)
                        .view(1, 1, -1)
                        .expand(llm_grid_t, llm_grid_h, -1)
                        .flatten()
                    )
                    llm_pos_ids_list.append(
                        torch.stack([t_index, h_index, w_index]) + text_len + st_idx
                    )
                    st = ed + llm_grid_t * llm_grid_h * llm_grid_w

                if st < len(input_tokens):
                    st_idx = (
                        llm_pos_ids_list[-1].max() + 1
                        if len(llm_pos_ids_list) > 0
                        else 0
                    )
                    text_len = len(input_tokens) - st
                    llm_pos_ids_list.append(
                        torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx
                    )

                llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
                position_ids[..., i, attention_mask[i] == 1] = llm_positions.to(
                    position_ids.device
                )
                mrope_position_deltas.append(
                    llm_positions.max() + 1 - len(total_input_ids[i])
                )
            mrope_position_deltas = torch.tensor(
                mrope_position_deltas, device=input_ids.device
            ).unsqueeze(1)
            return position_ids, mrope_position_deltas
        else:
            if attention_mask is not None:
                position_ids = attention_mask.long().cumsum(-1) - 1
                position_ids.masked_fill_(attention_mask == 0, 1)
                position_ids = (
                    position_ids.unsqueeze(0).expand(3, -1, -1).to(input_ids.device)
                )
                max_position_ids = position_ids.max(0, keepdim=False)[0].max(
                    -1, keepdim=True
                )[0]
                mrope_position_deltas = max_position_ids + 1 - attention_mask.shape[-1]
            else:
                position_ids = (
                    torch.arange(input_ids.shape[1], device=input_ids.device)
                    .view(1, 1, -1)
                    .expand(3, input_ids.shape[0], -1)
                )
                mrope_position_deltas = torch.zeros(
                    [input_ids.shape[0], 1],
                    device=input_ids.device,
                    dtype=input_ids.dtype,
                )

            return position_ids, mrope_position_deltas

    def prepare_inputs(self, data: Union[dict, tuple, list]):
        input_ids = data["input_ids"]

        attention_mask = None

        seq_length = input_ids.shape[1]

        assert self.embedding is not None, "Token embedding is not available."
        assert input_ids.shape[0] == 1, "Batch size should be 1 in inference mode."

        assert (
            seq_length <= self.input_sequence_length
        ), f"Input sequence length is too long. max input sequence length is {self.input_sequence_length} but got {seq_length}"
        if self.input_sequence_length > seq_length:
            padding_input_ids = torch.zeros(
                (1, self.input_sequence_length - seq_length), dtype=torch.long
            )
            padding_input_ids.fill_(self.pad_token_id)
            input_ids = torch.cat([input_ids, padding_input_ids], dim=-1)

        inputs_embeds = F.embedding(input_ids, self.embedding).cpu()
        n_image_tokens = torch.sum(input_ids == self.image_token_id).item()
        if n_image_tokens > 0:
            image_embeds = data["image_embeds"]
            n_image_features = image_embeds.shape[1]
            if n_image_tokens != n_image_features:
                raise ValueError(
                    f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
                )
            image_mask = (
                (input_ids == self.image_token_id)
                .unsqueeze(-1)
                .expand_as(inputs_embeds)
                .to(inputs_embeds.device)
            )
            image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

            deepstack_image_embeds = data["deepstack_image_embeds"]
            new_deepstack_image_embeds = list()
            for deepstack_image_embed in deepstack_image_embeds:
                new_deepstack_image_embed = torch.zeros_like(inputs_embeds).to(
                    deepstack_image_embed
                )
                new_deepstack_image_embed = new_deepstack_image_embed.masked_scatter(
                    image_mask.to(deepstack_image_embed.device), deepstack_image_embed
                )
                new_deepstack_image_embeds.append(new_deepstack_image_embed)
            deepstack_image_embed_0 = new_deepstack_image_embeds[0]
            deepstack_image_embed_1 = new_deepstack_image_embeds[1]
            deepstack_image_embed_2 = new_deepstack_image_embeds[2]
        else:
            deepstack_image_embed_0 = torch.zeros_like(inputs_embeds)
            deepstack_image_embed_1 = torch.zeros_like(inputs_embeds)
            deepstack_image_embed_2 = torch.zeros_like(inputs_embeds)

        past_seq_length = data["past_seq_length"]
        assert past_seq_length >= 0, "past_seq_length should be non-negative."

        if past_seq_length == 0:
            # prefill
            image_grid_thw = data["image_grid_thw"]
            video_grid_thw = None
            position_ids, rope_deltas = self.get_rope_index(
                input_ids, image_grid_thw, video_grid_thw, attention_mask
            )
            self.rope_deltas = rope_deltas
        else:
            assert (
                self.rope_deltas is not None
            ), f"rope_deltas is None, but past_seq_length is {past_seq_length}"
            batch_size, seq_length, _ = inputs_embeds.shape
            delta = past_seq_length + self.rope_deltas
            position_ids = torch.arange(seq_length, device=inputs_embeds.device)
            position_ids = position_ids.view(1, -1).expand(batch_size, -1)
            delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=0)
            position_ids = position_ids.add(delta)
            position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)

        time_position_ids = position_ids[0, 0].to(torch.int32)
        height_position_ids = position_ids[1, 0].to(torch.int32)
        width_position_ids = position_ids[2, 0].to(torch.int32)

        return (
            inputs_embeds,
            time_position_ids,
            height_position_ids,
            width_position_ids,
            torch.tensor([past_seq_length], dtype=torch.int32),
            torch.tensor([seq_length], dtype=torch.int32),
            deepstack_image_embed_0,
            deepstack_image_embed_1,
            deepstack_image_embed_2,
        )

    def run_prefill(self, data):
        input_ids = data["input_ids"]
        input_seq_len = input_ids.shape[-1]
        steps = (input_seq_len + self.prefill_len - 1) // self.prefill_len

        self.input_sequence_length = self.prefill_len * steps
        inputs = self.prepare_inputs(data)
        (
            inputs_embeds,
            time_position_ids,
            height_position_ids,
            width_position_ids,
            past_seq_length,
            _,
            deepstack_image_embed_0,
            deepstack_image_embed_1,
            deepstack_image_embed_2,
        ) = inputs
        current_length = inputs_embeds.shape[1]
        if current_length >= self.context_max_length:
            logger.error(
                f"Question long than {self.context_max_length}, please shorten it!"
            )
            sys.exit(1)
        for i in range(steps):
            start = i * self.prefill_len
            end = (i + 1) * self.prefill_len
            current_input_length = min(end, input_seq_len) - start
            self.prefill.set_input(
                self.prefill.get_input_name(0),
                inputs_embeds[:, start:end, :].detach().numpy(),
            )
            self.prefill.set_input(
                self.prefill.get_input_name(1),
                time_position_ids[start:end].detach().numpy(),
            )
            self.prefill.set_input(
                self.prefill.get_input_name(2),
                height_position_ids[start:end].detach().numpy(),
            )
            self.prefill.set_input(
                self.prefill.get_input_name(3),
                width_position_ids[start:end].detach().numpy(),
            )
            self.prefill.set_input(
                self.prefill.get_input_name(4),
                past_seq_length.detach().numpy(),
            )
            self.prefill.set_input(
                self.prefill.get_input_name(5),
                torch.tensor([current_input_length], dtype=torch.int32)
                .detach()
                .numpy(),
            )
            self.prefill.set_input(
                self.prefill.get_input_name(6),
                deepstack_image_embed_0[:, start:end, :].detach().numpy(),
            )
            self.prefill.set_input(
                self.prefill.get_input_name(7),
                deepstack_image_embed_1[:, start:end, :].detach().numpy(),
            )
            self.prefill.set_input(
                self.prefill.get_input_name(8),
                deepstack_image_embed_2[:, start:end, :].detach().numpy(),
            )
            start_time = time.time()
            self.prefill.run()
            self.prefill.sync()
            self.prefill_time += time.time() - start_time
            past_seq_length += current_input_length
            prefill_output = self.prefill.get_output(
                self.prefill.get_output_name(0)
            ).numpy()
        next_id = prefill_output.argmax(-1)
        return next_id, past_seq_length

    def run_visual(self, inputs):
        vit_input = inputs[0]
        self.vit_model.set_input(
            self.vit_model.get_input_name(0),
            vit_input.numpy(),
        )
        start_time = time.time()
        self.vit_model.run()
        self.vit_model.sync()
        self.vit_time += time.time() - start_time
        image_features = torch.Tensor(
            self.vit_model.get_output(self.vit_model.get_output_name(0)).numpy()
        )
        deepstack_image_feature_0 = torch.Tensor(
            self.vit_model.get_output(self.vit_model.get_output_name(1)).numpy()
        )
        deepstack_image_feature_1 = torch.Tensor(
            self.vit_model.get_output(self.vit_model.get_output_name(2)).numpy()
        )
        deepstack_image_feature_2 = torch.Tensor(
            self.vit_model.get_output(self.vit_model.get_output_name(3)).numpy()
        )
        return (
            image_features,
            deepstack_image_feature_0,
            deepstack_image_feature_1,
            deepstack_image_feature_2,
        )

    def chat_vit_prefill(self, image_path, prompt, system_prompt=None):
        self.generated_ids = []
        self.vit_time = 0
        self.decode_time = 0
        self.prefill_time = 0
        self.skip_tokens = 0
        self.slide_len = 10
        start_time = time.time()
        if image_path is not None:
            if self.resize_v1:
                pil_image = self.load_and_process_image(image_path)
            else:
                pil_image = self.load_and_process_image_v2(image_path)
        else:
            pil_image = None
        inputs = self.preprocess(prompt, pil_image, self.processor)
        inputs = inputs.to(self.device)

        if image_path is not None:
            visual_inputs = self.preprocess_visual(inputs)
            (
                image_features,
                deepstack_image_feature_0,
                deepstack_image_feature_1,
                deepstack_image_feature_2,
            ) = self.run_visual(visual_inputs)
        else:
            image_features = None
            deepstack_image_feature_0 = None
            deepstack_image_feature_1 = None
            deepstack_image_feature_2 = None
        deepstack_image_features = (
            deepstack_image_feature_0,
            deepstack_image_feature_1,
            deepstack_image_feature_2,
        )
        data_prefill = {
            "input_ids": inputs["input_ids"],
            "image_embeds": image_features,
            "deepstack_image_embeds": deepstack_image_features,
            "past_seq_length": 0,
            "image_grid_thw": inputs.get("image_grid_thw", None),
        }
        self.next_id, valid_length = self.run_prefill(data_prefill)
        self.generated_ids.append(self.next_id.item())
        ttft_time = time.time() - start_time
        self.last_response = self.processor.tokenizer.decode(
            self.generated_ids[-self.slide_len :]
        )
        self.all_response = self.last_response
        logger.success("response:")
        print("\033[1;95m{}".format(self.all_response), end="", flush=True)
        self.context_length = valid_length
        return ttft_time, inputs["input_ids"].shape[1]

    def chat_decoder(self):
        if self.context_length >= self.context_max_length:
            logger.error(
                f"Context length long than {self.context_max_length}, stop run decode model!"
            )
            return None
        self.input_sequence_length = 1
        data = {
            "input_ids": torch.Tensor(self.next_id).to(torch.int32),
            "past_seq_length": self.context_length,
        }
        inputs = self.prepare_inputs(data)
        (
            inputs_embeds,
            time_position_ids,
            height_position_ids,
            width_position_ids,
            past_seq_length,
            current_seq_length,
            deepstack_image_embed_0,
            deepstack_image_embed_1,
            deepstack_image_embed_2,
        ) = inputs
        self.decode.set_input(
            self.decode.get_input_name(0),
            inputs_embeds.detach().numpy(),
        )
        self.decode.set_input(
            self.decode.get_input_name(1),
            time_position_ids.detach().numpy(),
        )
        self.decode.set_input(
            self.decode.get_input_name(2),
            height_position_ids.detach().numpy(),
        )
        self.decode.set_input(
            self.decode.get_input_name(3),
            width_position_ids.detach().numpy(),
        )
        self.decode.set_input(
            self.decode.get_input_name(4),
            past_seq_length.detach().numpy(),
        )
        self.decode.set_input(
            self.decode.get_input_name(5),
            torch.tensor([current_seq_length], dtype=torch.int32).detach().numpy(),
        )
        self.decode.set_input(
            self.decode.get_input_name(6),
            deepstack_image_embed_0.detach().numpy(),
        )
        self.decode.set_input(
            self.decode.get_input_name(7),
            deepstack_image_embed_1.detach().numpy(),
        )
        self.decode.set_input(
            self.decode.get_input_name(8),
            deepstack_image_embed_2.detach().numpy(),
        )
        start_time = time.time()
        self.decode.run()
        self.decode.sync()
        self.decode_time += time.time() - start_time
        decoder_output = self.decode.get_output(self.decode.get_output_name(0)).numpy()
        self.next_id = self.samplingmanager.sample(decoder_output, self.generated_ids)
        self.generated_ids.append(self.next_id.item())
        if self.next_id.item() in self.eos_token_id:
            print(self.decode_response, end="", flush=True)
            self.all_response += self.decode_response
            return None
        self.context_length += 1
        self.decode_response = self.processor.tokenizer.decode(
            self.generated_ids[-(self.slide_len + 1) - self.skip_tokens :]
        )[len(self.last_response) :]
        if self.decode_response != "" and is_valid_char(ord(self.decode_response[-1])):
            print(self.decode_response, end="", flush=True)
            self.all_response += self.decode_response
            self.last_response = self.processor.tokenizer.decode(
                self.generated_ids[-self.slide_len :]
            )
            self.skip_tokens = 0
        else:
            self.skip_tokens += 1
        return self.decode_response


if __name__ == "__main__":
    args = get_args()
    qwen3vl = Qwen3VL(
        args.vit_path,
        args.prefill_path,
        args.decode_path,
        args.tokenizer_dir,
        args.embedding_path,
    )
    start_time = time.time()
    # image_dir = None
    image_dir = "../../../data/pic/beach.jpeg"
    image_num = 1 if image_dir else 0
    # prompt = "你好，你是谁?"
    prompt = "请描述图片内容。"
    logger.success("question:")
    print("\033[1;95m{}\033[0m".format(prompt))
    ttft_time, input_tokens = qwen3vl.chat_vit_prefill(image_dir, prompt=prompt)
    decode_count = 0
    while True:
        next_str = qwen3vl.chat_decoder()
        decode_count += 1
        if next_str is None:
            break
        # print(next_str, end="", flush=True)
    print("\033[0m")
    output_tokens = decode_count + 1
    vit_time = qwen3vl.vit_time
    prefill_time = qwen3vl.prefill_time
    decode_time = qwen3vl.decode_time
    total_time = time.time() - start_time
    logger.success(
        f"Total Images: {image_num}, Total Input: {input_tokens} tokens, Output {output_tokens} tokens, Vision Cost {vit_time * 1000:.3f} ms, Prefill Cost {prefill_time * 1000:.3f} ms, Decode Cost {decode_time * 1000:.3f} ms"
    )
    if image_num:
        logger.success(f"Vision Cost {vit_time / image_num * 1000:.3f} ms/image")
    logger.success(
        f"Prefill Speed: {input_tokens / prefill_time:.2f} tokens/s; Decode Speed: {(output_tokens - 1) / decode_time:.2f} tokens/s"
    )
    logger.success(f"TTFT (Time to First Token): {ttft_time* 1000:.3f} ms")
    logger.success(
        f"TPOT (Time Per Output Token): {decode_time * 1000 / (output_tokens - 1):.3f} ms/token"
    )
    logger.success(f"E2E Latency (End-to-End Latency): {total_time:.3f} seconds")
    logger.success(
        f"E2E TPS (End-to-End Tokens Per Second): {output_tokens / total_time:.2f} tokens/s"
    )
