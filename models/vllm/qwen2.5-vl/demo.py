import os
import time
import logging
import argparse
from loguru import logger
logging.basicConfig(level=logging.ERROR)
import warnings
warnings.simplefilter(action='ignore', category=UserWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)
import torch
import torch.nn.functional as F
import numpy as np
import tcim_lite as tcim
from PIL import Image
from processing_qwen2_5_vl import Qwen2_5_VLProcessor
from utils import get_rope_index, QRawToYuv

TOKENIZER_PATH = "qwen2.5-vl"
HOUMO_TARGET = os.getenv('HOUMO_TARGET', 'houmo')
EMBEDDING_PATH = os.path.join('output', HOUMO_TARGET, 'hmquant', 'quant_embedding.pt')

if HOUMO_TARGET == 'xh1':
    TARGET_TYPE = torch.long
elif HOUMO_TARGET == 'xh2':
    TARGET_TYPE = torch.float16

def get_args() -> argparse.Namespace:
    """Parse commandline."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model_dir',
        dest='model_dir',
        type=str,
        default=os.path.join('output', HOUMO_TARGET),
        help='houmo model dir',
    )
    parser.add_argument(
        '--prefill',
        dest='prefill_shape',
        type=list,
        default=(1, 256),
        help='prefill max length',
    )
    parser.add_argument(
        '--decode',
        dest='decode_length',
        type=int,
        default=2048,
        help='decode max length',
    )
    parser.add_argument(
        '--model_size',
        dest='model_size',
        type=str,
        default="7b",
        choices=["3b", "7b"],
        help='model size',
    )
    args = parser.parse_args()
    return args

class Qwen25VL:
    def __init__(
            self,
            model_dir,
            cache_len=2048,
            device="cpu",
            prefill_shape=(1, 256),
            window_size=112,
            spatial_merge_size=2,
            patch_size=14,
            model_size="3b", # 3B is 36 Blocks, 7B is 28 Blocks):
        ):
        self.processor = Qwen2_5_VLProcessor.from_pretrained(TOKENIZER_PATH + f"-{model_size}")
        self.cache_len = cache_len
        self.device = torch.device(device)
        self.prefill_shape = torch.Size(prefill_shape)
        self.prefill_len = self.prefill_shape.numel()

        self.window_size = window_size
        self.spatial_merge_size = spatial_merge_size
        self.patch_size = patch_size
        self.spatial_merge_unit = self.spatial_merge_size * self.spatial_merge_size
        if model_size == "3b":
            self.blocks = 36
        elif model_size == "7b":
            self.blocks = 28
        self.eos_token_id = [151645, 151643]
        # set mode
        self.rgb2yuv = QRawToYuv(input_color_type="RGB", toYUV_format="YUV444")
        weight_manager = tcim.runtime.WeightManager(0)
        option0 = tcim.runtime.Option(weight_manager)
        option1 = tcim.runtime.Option(weight_manager)
        option2 = tcim.runtime.Option(weight_manager)

        self.vit_model = tcim.runtime.load(os.path.join(model_dir, "qwen2.5-vl_visual.hmm"), option=option0)
        dummy_tensor_names = [f'model_layers_{i}_self_attn_kcache_input' for i in range(self.blocks)]
        dummy_tensor_names += [f'model_layers_{i}_self_attn_vcache_input' for i in range(self.blocks)]
        option2.set_dummy_tensors(dummy_tensor_names)
        self.prefill_model = tcim.runtime.load(os.path.join(model_dir, "qwen2.5-vl_prefill.hmm"), option=option1)
        self.decode_model = tcim.runtime.load(os.path.join(model_dir, "qwen2.5-vl_decode.hmm"), option=option2)
        for i in range(self.blocks):
            kcache = self.prefill_model.get_input(f"model_layers_{i}_self_attn_kcache_input")
            vcache = self.prefill_model.get_input(f"model_layers_{i}_self_attn_vcache_input")
            self.decode_model.set_input(f"model_layers_{i}_self_attn_kcache_input", kcache)
            self.decode_model.set_input(f"model_layers_{i}_self_attn_vcache_input", vcache)
        self.decode_model.set_input("current_length", np.array([1]).astype('int16'))
        self.embedding = torch.load(EMBEDDING_PATH, weights_only=False)
        if HOUMO_TARGET == 'xh2':
            self.embedding = self.embedding.weight
        self.hidden_dims = self.embedding.shape[-1]

    def create_template(self, prompt, image_dir):
        content_list = []
        if image_dir:
            for img_path in image_dir:
                content_list.append({"type": "image", "image": img_path})
        content_list.append({"type": "text", "text": prompt})
        messages = [
            {
                "role": "user",
                "content": content_list
            }
        ]
        return messages

    def preprocess(self, prompt, image_dir):
        messages = self.create_template(prompt, image_dir)
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        resized_image_inputs = None
        video_inputs = None
        if image_dir:
            resized_image_inputs = []
            try:
                from qwen_vl_utils import process_vision_info
                image_inputs, video_inputs = process_vision_info(messages)
                for image_input in image_inputs:
                    resized_image_input = image_input.resize((644, 364))
                    resized_image_inputs.append(resized_image_input)
            except:
                for content in messages[0]['content']:
                    if content['type']=='image':
                        image_input = Image.open(content['image'])
                        resized_image_input = image_input.resize((644, 364))
                        resized_image_inputs.append(resized_image_input)

        inputs = self.processor(
            text=[text],
            images=resized_image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        return inputs

    def get_window_index(self, grid_thw):
        window_index: list = []
        cu_window_seqlens: list = [0]
        window_index_id = 0
        vit_merger_window_size = (
            self.window_size // self.spatial_merge_size // self.patch_size
        )

        for grid_t, grid_h, grid_w in grid_thw:
            llm_grid_h, llm_grid_w = (
                grid_h // self.spatial_merge_size,
                grid_w // self.spatial_merge_size,
            )
            index = torch.arange(grid_t * llm_grid_h * llm_grid_w).reshape(
                grid_t, llm_grid_h, llm_grid_w
            )
            pad_h = vit_merger_window_size - llm_grid_h % vit_merger_window_size
            pad_w = vit_merger_window_size - llm_grid_w % vit_merger_window_size
            num_windows_h = (llm_grid_h + pad_h) // vit_merger_window_size
            num_windows_w = (llm_grid_w + pad_w) // vit_merger_window_size
            index_padded = F.pad(index, (0, pad_w, 0, pad_h), "constant", -100)
            index_padded = index_padded.reshape(
                grid_t,
                num_windows_h,
                vit_merger_window_size,
                num_windows_w,
                vit_merger_window_size,
            )
            index_padded = index_padded.permute(0, 1, 3, 2, 4).reshape(
                grid_t,
                num_windows_h * num_windows_w,
                vit_merger_window_size,
                vit_merger_window_size,
            )
            seqlens = (index_padded != -100).sum([2, 3]).reshape(-1)
            index_padded = index_padded.reshape(-1)
            index_new = index_padded[index_padded != -100]
            window_index.append(index_new + window_index_id)
            cu_seqlens_tmp = (
                seqlens.cumsum(0) * self.spatial_merge_unit + cu_window_seqlens[-1]
            )
            cu_window_seqlens.extend(cu_seqlens_tmp.tolist())
            window_index_id += (grid_t * llm_grid_h * llm_grid_w).item()
        window_index = torch.cat(window_index, dim=0)

        return window_index, cu_window_seqlens

    def preprocess_visual(self, inputs): 
        visual_inputs = dict()
        visual_inputs["hidden_states"] = inputs["hm_pixel_values"].cpu()
        window_indexes = []
        window_masks = []
        for batch in range(inputs["image_grid_thw"].shape[0]):
            window_index, cu_window_seqlens = self.get_window_index(inputs["image_grid_thw"][batch].unsqueeze(0))
            cu_window_seqlens = torch.tensor(cu_window_seqlens, device=self.device, dtype=torch.int32)
            cu_window_seqlens = torch.unique_consecutive(cu_window_seqlens)
            seq_len = cu_window_seqlens[-1]
            attention_mask = torch.full(
                [1, seq_len, seq_len],
                torch.iinfo(torch.int16).min,
                device=inputs["hm_pixel_values"].device,
                dtype=torch.int16,
            )
            for i in range(1, len(cu_window_seqlens)):
                attention_mask[
                    ...,
                    cu_window_seqlens[i - 1] : cu_window_seqlens[i],
                    cu_window_seqlens[i - 1] : cu_window_seqlens[i],
                ] = 0

            window_indexes.append(window_index.to(self.device))
            window_masks.append(attention_mask.to(self.device))
        visual_inputs["window_index"] = torch.stack(window_indexes)
        visual_inputs["window_mask"] = torch.stack(window_masks)
        if HOUMO_TARGET == 'xh1':
            visual_inputs["hidden_states"] = self.rgb2yuv(visual_inputs["hidden_states"].cpu()).long()
        return visual_inputs

    if HOUMO_TARGET == 'xh1':
        def run_visual(self, inputs):
            vit_model_outputs = list()
            for batch in range(inputs["hidden_states"] .shape[0]):
                self.vit_model.set_input(self.vit_model.get_input_name(0), inputs["hidden_states"].numpy().astype(np.uint8))
                self.vit_model.set_input(self.vit_model.get_input_name(1), inputs["window_index"].numpy().astype(np.int32))
                self.vit_model.set_input(self.vit_model.get_input_name(2), inputs["window_mask"].numpy().astype(np.int16))
                self.vit_model.run()
                self.vit_model.sync()
                # vit_model_output = self.vit_model.get_output("output").numpy().astype(np.int16)
                vit_model_output = self.vit_model.get_output(self.vit_model.get_output_name(0)).numpy().astype(np.int16)
                vit_model_outputs.append(torch.tensor(vit_model_output))
            #del self.vit_model
            return torch.cat(vit_model_outputs, dim=0)
    elif HOUMO_TARGET == 'xh2':
        def run_visual(self, inputs):
            vit_model_outputs = list()
            for batch in range(inputs["hidden_states"] .shape[0]):
                self.vit_model.set_input(self.vit_model.get_input_name(0), inputs["hidden_states"][batch].unsqueeze(0).numpy().astype(np.float16))
                self.vit_model.set_input(self.vit_model.get_input_name(1), inputs["window_index"][batch].numpy().astype(np.int32))
                self.vit_model.set_input(self.vit_model.get_input_name(2), inputs["window_mask"][batch].numpy().astype(np.float16))
                self.vit_model.run()
                self.vit_model.sync()
                # vit_model_output = self.vit_model.get_output("output").numpy().astype(np.int16)
                vit_model_output = self.vit_model.get_output(self.vit_model.get_output_name(0)).numpy().astype(np.float16)
                vit_model_outputs.append(torch.tensor(vit_model_output))
                #del self.vit_model
            return torch.cat(vit_model_outputs, dim=0)

    def preprocess_prefill(self, inputs, image_features):
        image_grid_thw = None
        input_ids = inputs["input_ids"].cpu()
        inputs_embeds = F.embedding(input_ids, self.embedding).cpu()
        mask = input_ids == 151655  # <image> token id
        mask_unsqueezed = mask.unsqueeze(-1)
        mask_expanded = mask_unsqueezed.expand_as(inputs_embeds)
        image_mask = mask_expanded
        if image_features is not None:
            image_features = image_features.type(TARGET_TYPE)
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_features)
            image_grid_thw = inputs["image_grid_thw"]

        position_ids, rope_deltas = get_rope_index(
            input_ids,
            image_grid_thw,
            None, # video_grid_thw is None
            None,
            inputs["attention_mask"].cpu(),
        )

        time_position_ids = position_ids[0][0]
        hight_position_ids = position_ids[1][0]
        width_position_ids = position_ids[2][0]
        return inputs_embeds, time_position_ids, hight_position_ids, width_position_ids, rope_deltas
    def create_prefill_inputs(self, inputs_embeds, time_position_ids, hight_position_ids, width_position_ids, pre_gen_idx):
        x = inputs_embeds[
            :,
            pre_gen_idx
            * self.prefill_len : (pre_gen_idx + 1)
            * self.prefill_len,
        ]
        x_time = time_position_ids[
            pre_gen_idx
            * self.prefill_len : (pre_gen_idx + 1)
            * self.prefill_len
        ]
        x_hight = hight_position_ids[
            pre_gen_idx
            * self.prefill_len : (pre_gen_idx + 1)
            * self.prefill_len
        ]
        x_width = width_position_ids[
            pre_gen_idx
            * self.prefill_len : (pre_gen_idx + 1)
            * self.prefill_len
        ]
        p_current_length = torch.tensor([self.prefill_len])
        p_valid_length = (p_current_length * pre_gen_idx).to(TARGET_TYPE)
        prefill_inputs = dict(
            input_1=x,
            valid_length=p_valid_length,
            current_length=p_current_length,
            time_position_ids=x_time,
            hight_position_ids=x_hight,
            width_position_ids=x_width,
        )
        return prefill_inputs

    def run_prefill(self, inputs_embeds, time_position_ids, hight_position_ids, width_position_ids):
        current_length = inputs_embeds.shape[1]
        if current_length > self.prefill_len:
            pre_gen_nums = current_length // self.prefill_len
            for pre_gen_idx in range(pre_gen_nums):
                prefill_inputs = self.create_prefill_inputs(inputs_embeds, time_position_ids, hight_position_ids, width_position_ids, pre_gen_idx)
                self.prefill_model.set_input(self.prefill_model.get_input_name(0), prefill_inputs['input_1'].detach().numpy())
                self.prefill_model.set_input(self.prefill_model.get_input_name(1), prefill_inputs['time_position_ids'].detach().numpy())
                self.prefill_model.set_input(self.prefill_model.get_input_name(2), prefill_inputs['hight_position_ids'].detach().numpy())
                self.prefill_model.set_input(self.prefill_model.get_input_name(3), prefill_inputs['width_position_ids'].detach().numpy())
                self.prefill_model.set_input(self.prefill_model.get_input_name(4), prefill_inputs['valid_length'].detach().numpy())
                self.prefill_model.set_input(self.prefill_model.get_input_name(5), prefill_inputs['current_length'].detach().numpy())
                self.prefill_model.run()
                self.prefill_model.sync()
                prefill_output = self.prefill_model.get_output(self.prefill_model.get_output_name(0))
        else:
            pre_gen_nums = 0

        current_length = current_length % self.prefill_len
        prefill_shape = list(self.prefill_shape)
        prefill_shape.append(self.hidden_dims)
        x = torch.zeros(prefill_shape, dtype=TARGET_TYPE)
        x[:, :current_length] = inputs_embeds[:, -current_length:]

        x_time = torch.zeros(self.prefill_len, dtype=TARGET_TYPE)
        x_hight = torch.zeros(self.prefill_len, dtype=TARGET_TYPE)
        x_width = torch.zeros(self.prefill_len, dtype=TARGET_TYPE)
        x_time[:current_length] = time_position_ids[-current_length:]
        x_hight[:current_length] = hight_position_ids[-current_length:]
        x_width[:current_length] = width_position_ids[-current_length:]
        current_length = torch.tensor([current_length])
        valid_length = (torch.tensor([self.prefill_len]) * pre_gen_nums).to(TARGET_TYPE)
        prefill_inputs = dict(
            input_1=x,
            valid_length=valid_length,
            current_length=current_length,
            time_position_ids=x_time,
            hight_position_ids=x_hight,
            width_position_ids=x_width,
        )
        self.prefill_model.set_input(self.prefill_model.get_input_name(0), prefill_inputs['input_1'].detach().numpy())
        self.prefill_model.set_input(self.prefill_model.get_input_name(1), prefill_inputs['time_position_ids'].detach().numpy())
        self.prefill_model.set_input(self.prefill_model.get_input_name(2), prefill_inputs['hight_position_ids'].detach().numpy())
        self.prefill_model.set_input(self.prefill_model.get_input_name(3), prefill_inputs['width_position_ids'].detach().numpy())
        self.prefill_model.set_input(self.prefill_model.get_input_name(4), prefill_inputs['valid_length'].detach().numpy())
        self.prefill_model.set_input(self.prefill_model.get_input_name(5), prefill_inputs['current_length'].detach().numpy())
        self.prefill_model.run()
        self.prefill_model.sync()
        prefill_output = self.prefill_model.get_output(self.prefill_model.get_output_name(0))
        next_id = prefill_output.numpy().argmax(-1)
        return prefill_output, next_id, valid_length, current_length

    def chat_vit_prefill(
            self,
            image_dir,
            prompt,
            system_prompt=None
    ):
        image_features=None
        inputs = self.preprocess(prompt, image_dir)
        if image_dir != None:
            visual_inputs = self.preprocess_visual(inputs)
            image_features = self.run_visual(visual_inputs)
        inputs_embeds, time_position_ids, hight_position_ids, width_position_ids, self.rope_deltas = self.preprocess_prefill(inputs, image_features)
        prefill_output, self.next_id, valid_length, current_length = self.run_prefill(inputs_embeds, time_position_ids, hight_position_ids, width_position_ids)
        next_str = self.processor.tokenizer.decode(torch.tensor(self.next_id.item()))
        print(f'\033[1;95m{next_str}', end='', flush=True)
        self.context_length = valid_length + current_length + 1

    def chat_decoder(self):
        if self.context_length >= self.cache_len:
            return None
        decoder_pids = self.context_length + self.rope_deltas.item() - 1
        x = F.embedding(torch.from_numpy(self.next_id).unsqueeze(0), self.embedding)
        decoder_inputs = dict(
            input_1=x,
            valid_length=torch.tensor(self.context_length - 1),
            current_length=torch.tensor([1]).long(),
            time_position_ids=torch.tensor(decoder_pids),
            hight_position_ids=torch.tensor(decoder_pids),
            width_position_ids=torch.tensor(decoder_pids),
        )
        if HOUMO_TARGET == "xh2":
            decoder_inputs['input_1'] = decoder_inputs['input_1'].squeeze(0)
        self.decode_model.set_input(self.decode_model.get_input_name(0), decoder_inputs['input_1'].detach().numpy())
        self.decode_model.set_input(self.decode_model.get_input_name(1), decoder_inputs['time_position_ids'].detach().numpy())
        self.decode_model.set_input(self.decode_model.get_input_name(2), decoder_inputs['hight_position_ids'].detach().numpy())
        self.decode_model.set_input(self.decode_model.get_input_name(3), decoder_inputs['width_position_ids'].detach().numpy())
        self.decode_model.set_input(self.decode_model.get_input_name(4), decoder_inputs['valid_length'].detach().numpy())
        self.decode_model.set_input(self.decode_model.get_input_name(5), decoder_inputs['current_length'].detach().numpy())
        self.decode_model.run()
        self.decode_model.sync()
        decoder_output = self.decode_model.get_output(self.decode_model.get_output_name(0))
        self.next_id = decoder_output.numpy().argmax(-1)
        if self.next_id.item() in self.eos_token_id:
            return None
        next_str = self.processor.tokenizer.decode(self.next_id.item())
        self.context_length += 1
        return next_str

if __name__ == "__main__":
    args = get_args()
    qwen25vl = Qwen25VL(model_dir=args.model_dir, prefill_shape=args.prefill_shape, cache_len=args.decode_length, model_size=args.model_size)
    # image_dir = None
    image_dir = ["../../../data/pic/beach.jpeg", "../../../data/pic/lane.jpg"]
    start_time = time.time()
    # qwen25vl.chat_vit_prefill(image_dir, prompt='你好，你是谁。')
    qwen25vl.chat_vit_prefill(image_dir, prompt='请描述图片内容。')
    visual_prefill_time = time.time() - start_time
    decode_count = 0
    while(True):
        next_str = qwen25vl.chat_decoder()
        decode_count += 1
        if next_str is None:
            break
        print(next_str, end='', flush=True)
    print('\033[0m')
    tokens = decode_count + 1
    decode_time = time.time() - start_time - visual_prefill_time
    total_time = time.time() - start_time
    logger.success(f"total: {tokens} tokens, cost {total_time:.3f} s")
    logger.success(f"visual + prefill time: {visual_prefill_time * 1000:.3f} ms, {1 / visual_prefill_time:.2f} tokens/s")
    decode_latency = decode_time * 1000 / (tokens - 1)
    logger.success(f"decode average time: {decode_latency:.3f} ms, {1000 / decode_latency:.2f} tokens/s")
    res_latency = total_time * 1000 / tokens
    logger.success(f"end2end average time: {res_latency:.3f} ms, {1000 / res_latency:.2f} tokens/s")