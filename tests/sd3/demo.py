# Copyright 2025 HOUMO AI
#
# File: demo.py
# Description:
#   Demo script for Stable Diffusion 3 inference using compiled HMM models.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

import os
import numpy as np
import tcim_lite
import torch
import torch.nn as nn
from diffusers import StableDiffusion3Pipeline
from transformers.models.clip.modeling_clip import CLIPTextModelOutput
from loguru import logger
import time
import argparse


HOUMO_TARGET = os.getenv("HOUMO_TARGET", "houmo")


class DiT(nn.Module):
    def __init__(self, DiT, model_path):
        super().__init__()
        self.config = DiT.config
        self.dtype = torch.float32
        self.profile = {}
        self.profile["dit_infer"] = 0
        start = time.time()
        self.module = tcim_lite.runtime.Module.load(model_path)
        cost = time.time() - start
        print(f"module {model_path} loaded success. cost {cost*1000:.3f} ms.")
        input_num = self.module.get_num_inputs()
        print("input_num:", input_num)
        for index in range(input_num):
            input_name = self.module.get_input_name(index)
            input_info = self.module.get_input_info(input_name)
            print(
                f"input[{input_name}] shape = {input_info.shape}, dtype = {input_info.dtype}, format = {input_info.format.name}"
            )
        output_num = self.module.get_num_outputs()
        print("output_num:", output_num)
        for index in range(output_num):
            output_name = self.module.get_output_name(index)
            output_info = self.module.get_output_info(output_name)
            print(
                f"output[{output_name}] shape = {output_info.shape}, dtype = {output_info.dtype}, format = {output_info.format.name}"
            )
        self.step = 0

    def forward(
        self,
        hidden_states,
        encoder_hidden_states,
        pooled_projections,
        timestep,
        **kwargs,
    ):
        start = time.time()
        hidden_states_input = hidden_states.to("cpu").numpy().astype(np.float16)
        encoder_hidden_states_input = (
            encoder_hidden_states.to("cpu").numpy().astype(np.float16)
        )
        pooled_projections_input = (
            pooled_projections.to("cpu").numpy().astype(np.float16)
        )
        timestep_input = timestep.to("cpu").numpy().astype(np.float16)
        self.module.set_input("hidden_states.hmcc.format", hidden_states_input)
        self.module.set_input(
            "encoder_hidden_states.hmcc.format", encoder_hidden_states_input
        )
        self.module.set_input(
            "pooled_projections.hmcc.format", pooled_projections_input
        )
        self.module.set_input("timestep.hmcc.format", timestep_input)
        self.module.run()
        self.module.sync()
        output_name = self.module.get_output_name(0)
        output = self.module.get_output(output_name).numpy().astype(np.float32)
        cost = time.time() - start
        self.profile["dit_infer"] += cost
        self.step += 1
        return (torch.from_numpy(output),)

    def prepare_inputs(self, data):
        """
        获取模型输入
        """
        return [
            data["hidden_states"],
            data["encoder_hidden_states"],
            data["pooled_projections"],
            data["timestep"],
        ]


class Clip(nn.Module):
    def __init__(self, input_shape, model_path):
        super().__init__()
        self.input_shape = input_shape
        self.dtype = torch.float32
        self.device = "cpu"
        self.profile = {}
        self.profile["clip_infer"] = 0
        start = time.time()
        self.module = tcim_lite.runtime.Module.load(model_path)
        cost = time.time() - start
        print(f"module {model_path} loaded success. cost {cost*1000:.3f} ms.")
        input_num = self.module.get_num_inputs()
        print("input_num:", input_num)
        for index in range(input_num):
            input_name = self.module.get_input_name(index)
            input_info = self.module.get_input_info(input_name)
            print(
                f"input[{input_name}] shape = {input_info.shape}, dtype = {input_info.dtype}, format = {input_info.format.name}"
            )
        output_num = self.module.get_num_outputs()
        print("output_num:", output_num)
        for index in range(output_num):
            output_name = self.module.get_output_name(index)
            output_info = self.module.get_output_info(output_name)
            print(
                f"output[{output_name}] shape = {output_info.shape}, dtype = {output_info.dtype}, format = {output_info.format.name}"
            )

    def forward(
        self,
        inputs,
        data_samples=None,
        **kwargs,
    ):
        start = time.time()
        inputs = inputs.to("cpu").numpy().astype(np.int32)
        self.module.set_input("input_ids.hmcc.format", inputs)
        self.module.run()
        self.module.sync()
        text_embeds = self.module.get_output("text_embeds").numpy()
        hidden_states = self.module.get_output("hidden_states").numpy()
        cost = time.time() - start
        self.profile["clip_infer"] += cost
        text_embeds = torch.from_numpy(text_embeds).to(self.dtype).to(self.device)
        hidden_states = torch.from_numpy(hidden_states).to(self.dtype).to(self.device)
        outputs = CLIPTextModelOutput(
            text_embeds=text_embeds,
            last_hidden_state=None,
            hidden_states=[hidden_states, 0],
            attentions=None,
        )
        return outputs


class T5(nn.Module):
    def __init__(self, model_path):
        super().__init__()
        self.dtype = torch.float32
        self.device = "cpu"
        self.profile = {}
        self.profile["t5_infer"] = 0
        start = time.time()
        self.module = tcim_lite.runtime.Module.load(model_path)
        cost = time.time() - start
        print(f"module {model_path} loaded success. cost {cost*1000:.3f} ms.")
        input_num = self.module.get_num_inputs()
        print("input_num:", input_num)
        for index in range(input_num):
            input_name = self.module.get_input_name(index)
            input_info = self.module.get_input_info(input_name)
            print(
                f"input[{input_name}] shape = {input_info.shape}, dtype = {input_info.dtype}, format = {input_info.format.name}"
            )
        output_num = self.module.get_num_outputs()
        print("output_num:", output_num)
        for index in range(output_num):
            output_name = self.module.get_output_name(index)
            output_info = self.module.get_output_info(output_name)
            print(
                f"output[{output_name}] shape = {output_info.shape}, dtype = {output_info.dtype}, format = {output_info.format.name}"
            )

    def forward(
        self,
        inputs,
        data_samples=None,
        **kwargs,
    ):
        start = time.time()
        inputs = inputs.to("cpu").numpy().astype(np.int32)
        self.module.set_input("input_ids.hmcc.format", inputs)
        self.module.run()
        self.module.sync()
        text_embeds = self.module.get_output("text_embeds").numpy()
        cost = time.time() - start
        self.profile["t5_infer"] += cost
        text_embeds = torch.from_numpy(text_embeds).to(self.dtype).to(self.device)
        return (text_embeds,)


class Vae(nn.Module):
    def __init__(self, Vae, model_path):
        super().__init__()
        self.dtype = torch.float32
        self.config = Vae.config
        self.profile = {}
        self.profile["vae_infer"] = 0
        start = time.time()
        self.module = tcim_lite.runtime.Module.load(model_path)
        cost = time.time() - start
        print(f"module {model_path} loaded success. cost {cost*1000:.3f} ms.")
        input_num = self.module.get_num_inputs()
        print("input_num:", input_num)
        for index in range(input_num):
            input_name = self.module.get_input_name(index)
            input_info = self.module.get_input_info(input_name)
            print(
                f"input[{input_name}] shape = {input_info.shape}, dtype = {input_info.dtype}, format = {input_info.format.name}"
            )
        output_num = self.module.get_num_outputs()
        print("output_num:", output_num)
        for index in range(output_num):
            output_name = self.module.get_output_name(index)
            output_info = self.module.get_output_info(output_name)
            print(
                f"output[{output_name}] shape = {output_info.shape}, dtype = {output_info.dtype}, format = {output_info.format.name}"
            )

    def decode(self, inputs, **kwargs):
        start = time.time()
        hidden_states_input = inputs.to("cpu").numpy().astype(np.float16)
        self.module.set_input("input.hmcc.format", hidden_states_input)
        self.module.run()
        self.module.sync()
        output_name = self.module.get_output_name(0)
        output = self.module.get_output(output_name).numpy().astype(np.float32)
        cost = time.time() - start
        self.profile["vae_infer"] += cost
        return (torch.from_numpy(output),)


def get_args() -> argparse.Namespace:
    """Parse commandline."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path",
        dest="model_path",
        type=str,
        default=f"output/{HOUMO_TARGET}",
        help="path to hmm model",
    )
    parser.add_argument(
        "--sd3_ckpt",
        dest="sd3_ckpt",
        type=str,
        default="stable-diffusion-3-medium-diffusers",
        help="path to sd3_ckpt",
    )
    parser.add_argument(
        "--test_num",
        dest="test_num",
        type=int,
        default=1,
        help="batch size",
    )
    parser.add_argument(
        "--nstep",
        dest="nstep",
        type=int,
        default=10,
        help="num_inference_steps",
    )
    parser.add_argument(
        "--prompt",
        dest="prompt",
        type=str,
        default=None,
        help="user prompt",
    )
    parser.add_argument(
        "--neg_prompt",
        dest="neg_prompt",
        type=str,
        default="",
        help="user negative prompt",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    logger.info(args)
    test_num = args.test_num
    model_path = args.model_path
    sd3_ckpt = args.sd3_ckpt
    num_inference_steps = args.nstep
    prompt = args.prompt
    neg_prompt = args.neg_prompt

    save_results = "demo_results"
    if not os.path.exists(save_results):
        os.makedirs(save_results)

    pipe = StableDiffusion3Pipeline.from_pretrained(sd3_ckpt, use_safetensors=True)
    pipe.to("cpu")

    clip_b_path = os.path.join(model_path, "clip.hmm")
    clip_b_model = Clip(768, clip_b_path)

    clip_l_path = os.path.join(model_path, "clip_l.hmm")
    clip_l_model = Clip(1280, clip_l_path)

    t5_path = os.path.join(model_path, "t5.hmm")
    t5_model = T5(t5_path)

    hmdit_path = os.path.join(model_path, "mmdit.hmm")
    dit_model = DiT(pipe.transformer, hmdit_path)

    vae_path = os.path.join(model_path, "vae.hmm")
    vae_model = Vae(pipe.vae, vae_path)

    pipe.text_encoder = clip_b_model
    pipe.text_encoder_2 = clip_l_model
    pipe.text_encoder_3 = t5_model
    pipe.transformer = dit_model
    pipe.vae = vae_model

    if prompt is not None:
        prompts = [prompt for _ in range(test_num)]
        negative_prompts = [[neg_prompt] for _ in range(test_num)]
    else:
        prompts = [str() for _ in range(test_num)]
        negative_prompts = [[str()] for _ in range(test_num)]

        prompts_list = [
            # 一位穿着粉色连衣裙的可爱年轻女孩在花园里微笑。照片级真实感，8K。
            "A cute young girl in a pink dress, smiling in a garden. Photo - realistic, 8k.",
            # 山间的瀑布，绿树和清澈的水。照片级真实，8K。
            "A waterfall in the mountains, green trees and clear water. Photo - real, 8k.",
            # 一只威严的龙在神奇森林中的场景，包含龙的特征、环境元素等。
            "A majestic dragon with iridescent scales, spreading its wings in a magical forest, glowing mushrooms on the ground, fire in its mouth, highly detailed.",
        ]

        negative_prompts_list = [
            # 避免生成模糊、低质量、变形、颜色难看、构图不佳以及卡通化的图像。
            [
                "Blurry, low - quality, distorted, ugly colors, bad composition, cartoonish."
            ],
            # 避免生成低质量，虚假质感的水。
            ["Poor details, fake - looking water."],
            # 防止生成单色、扁平、背景单调以及解剖结构错误的奇幻生物图像。
            ["Monochrome, flat, boring background, bad anatomy."],
        ]

        for i in range(test_num):
            prompts[i] = prompts_list[i % 3]
            negative_prompts[i] = negative_prompts_list[i % 3]

    for i in range(test_num):
        start = time.time()
        print(prompts[i])
        print(negative_prompts[i])
        image = pipe(
            prompt=prompts[i],
            negative_prompt=negative_prompts[i],
            guidance_scale=7.0,
            num_inference_steps=num_inference_steps,
            width=512,
            height=512,
        ).images[0]
        cost = time.time() - start
        save_path = os.path.join(save_results, f"{i}.jpg")
        image.save(save_path)
        clip_b_time = clip_b_model.profile["clip_infer"]
        clip_l_time = clip_l_model.profile["clip_infer"]
        t5_time = t5_model.profile["t5_infer"]
        dit_time = dit_model.profile["dit_infer"]
        vae_time = vae_model.profile["vae_infer"]
        avg_dit_time = dit_time / num_inference_steps
        logger.success(f"demo results saved in {save_path}, cost {cost*1000:.3f} ms.")
        logger.success(f"clip b infer cost {clip_b_time*1000:.3f} ms.")
        logger.success(f"clip l infer cost {clip_l_time*1000:.3f} ms.")
        logger.success(f"t5 infer cost {t5_time*1000:.3f} ms.")
        logger.success(
            f"dit infer {num_inference_steps} steps cost {dit_time*1000:.3f} ms, average {avg_dit_time*1000:.3f} ms."
        )
        logger.success(f"vae infer cost {vae_time*1000:.3f} ms.")
