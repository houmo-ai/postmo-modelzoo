import os
import numpy as np
import tcim_lite
import torch
from diffusers import StableDiffusionXLPipeline
from diffusers import TCDScheduler
from loguru import logger
import time
import argparse


HOUMO_TARGET = os.getenv("HOUMO_TARGET")
assert HOUMO_TARGET == "xh1", "Only support HOUMO_TARGET: xh1."


def quantize_to_int16(tensor, scale, zero_point=0):
    quantized_tensor = np.clip(
        np.round(tensor / scale + zero_point),
        -32768,
        32767,
    ).astype(np.int16)
    return quantized_tensor


def quantize_to_int8(tensor, scale, zero_point=0):
    quantized_tensor = np.clip(
        np.round(tensor / scale + zero_point),
        -128,
        127,
    ).astype(np.int8)
    return quantized_tensor


def dequantize_from_int16(quantized_tensor, scale, zero_point=0):
    dequantized_tensor = (
        quantized_tensor.astype(
            np.float32,
        )
        - zero_point
    ) * scale
    return dequantized_tensor


class HmUnet(torch.nn.Module):
    def __init__(self, UNET, model_path):
        super().__init__()
        self.add_embedding = UNET.add_embedding
        self.encoder_hid_proj = UNET.encoder_hid_proj
        self.conv_in = UNET.conv_in
        self.down_blocks = UNET.down_blocks
        self.mid_block = UNET.mid_block
        self.up_blocks = UNET.up_blocks
        self.conv_norm_out = UNET.conv_norm_out  # groupnorm
        self.conv_act = UNET.conv_act  # SILU
        self.conv_out = UNET.conv_out
        self.get_time_embed = UNET.get_time_embed
        self.time_embedding = UNET.time_embedding
        self.get_class_embed = UNET.get_class_embed
        self.config = UNET.config
        self.get_aug_embed = UNET.get_aug_embed
        self.time_embed_act = UNET.time_embed_act
        self.process_encoder_hidden_states = UNET.process_encoder_hidden_states
        self.dtype = torch.float32
        self.profile = {}
        self.profile["unet_infer"] = 0
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
                "input[{}] shape = {}, dtype = {}, format = {}".format(
                    input_name,
                    input_info.shape,
                    input_info.dtype,
                    input_info.format.name,
                )
            )
        output_num = self.module.get_num_outputs()
        print("output_num:", output_num)
        for index in range(output_num):
            output_name = self.module.get_output_name(index)
            output_info = self.module.get_output_info(output_name)
            print(
                "output[{}] shape = {}, dtype = {}, format = {}".format(
                    output_name,
                    output_info.shape,
                    output_info.dtype,
                    output_info.format.name,
                )
            )

    def get_embeddings(
        self,
        sample,
        timestep,
        encoder_hidden_states,
        added_cond_kwargs,
    ):
        t_emb = self.get_time_embed(
            sample=sample,
            timestep=timestep,
        )  # [2, 320]
        emb = self.time_embedding(t_emb, None)  # [2, 1280]
        aug_emb = None
        class_emb = self.get_class_embed(
            sample=sample,
            class_labels=None,
        )  # none
        if class_emb is not None:
            if self.config.class_embeddings_concat:
                emb = torch.cat([emb, class_emb], dim=-1)
            else:
                emb = emb + class_emb
        aug_emb = self.get_aug_embed(
            emb=emb,
            encoder_hidden_states=encoder_hidden_states,
            added_cond_kwargs=added_cond_kwargs,
        )  # [2, 1280]      pooled_emb + embding( add_time )
        if self.config.addition_embed_type == 'image_hint':
            aug_emb, hint = aug_emb
            sample = torch.cat([sample, hint], dim=1)
        emb = emb + aug_emb if aug_emb is not None else emb
        if self.time_embed_act is not None:
            emb = self.time_embed_act(emb)
        encoder_hidden_states = self.process_encoder_hidden_states(
            encoder_hidden_states=encoder_hidden_states,
            added_cond_kwargs=added_cond_kwargs,
        )
        return emb, encoder_hidden_states

    def forward(
        self,
        inp_sample,
        inp_time_emb,
        encoder_hidden_states,
        timestep_cond,
        added_cond_kwargs,
        cross_attention_kwargs,
        return_dict: bool = True,
        do_classifier_free_guidance=True,
    ):
        emb, encoder_hidden_states_processed = self.get_embeddings(
            inp_sample,
            inp_time_emb,
            encoder_hidden_states,
            added_cond_kwargs,
        )
        start = time.time()
        input = quantize_to_int16(inp_sample.cpu().numpy(), 0.00015613020514138043)
        input_9 = quantize_to_int8(emb.cpu().numpy(), 0.06447089463472366)
        encoder_hidden_states = quantize_to_int16(
            encoder_hidden_states_processed[0:1].cpu().numpy(), 0.026054341346025467
        )
        self.module.set_input('input', input)
        self.module.set_input('input.9', input_9)
        self.module.set_input('encoder_hidden_states', encoder_hidden_states)
        self.module.run()
        self.module.sync()
        output_name = self.module.get_output_name(0)
        output = self.module.get_output(output_name).numpy()
        output = output.reshape(1, 4, 128, 128).astype(np.float32)
        output[0][0] = dequantize_from_int16(output[0][0], 0.00012981216423213482)
        output[0][1] = dequantize_from_int16(output[0][1], 0.00012109786621294916)
        output[0][2] = dequantize_from_int16(output[0][2], 0.0001298204151680693)
        output[0][3] = dequantize_from_int16(output[0][3], 0.00015667281695641577)
        cost = time.time() - start
        self.profile["unet_infer"] += cost
        return torch.from_numpy(output).to(inp_sample.device)


def get_args() -> argparse.Namespace:
    """Parse commandline."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model_path',
        dest='model_path',
        type=str,
        default=f'output/{HOUMO_TARGET}/sd_unet.hmm',
        help='path to hmm model',
    )
    parser.add_argument(
        '--sdxl_ckpt',
        dest='sdxl_ckpt',
        type=str,
        default='stable-diffusion-xl-base-1.0',
        help='path to sdxl_ckpt',
    )
    parser.add_argument(
        '--lora_weights',
        dest='lora_weights',
        type=str,
        default='TCD-SDXL-LoRA',
        help='path to lora_weights',
    )
    parser.add_argument(
        '--test_num',
        dest='test_num',
        type=int,
        default=3,
        help='batch size',
    )
    parser.add_argument(
        '--nstep',
        dest='nstep',
        type=int,
        default=10,
        help='num_inference_steps',
    )
    parser.add_argument(
        '--eta',
        dest='eta',
        type=int,
        default=0.3,
        help='Sampling random noise eta in [0,1]',
    )
    parser.add_argument(
        '--prompt',
        dest='prompt',
        type=str,
        default=None,
        help='user prompt',
    )
    parser.add_argument(
        '--neg_prompt',
        dest='neg_prompt',
        type=str,
        default='',
        help='user negative prompt',
    )
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_args()
    logger.info(args)
    test_num = args.test_num
    model_path = args.model_path
    sdxl_ckpt = args.sdxl_ckpt
    lora_weights = args.lora_weights
    num_inference_steps = args.nstep
    eta = args.eta
    prompt = args.prompt
    neg_prompt = args.neg_prompt

    save_results = "demo_results"
    if not os.path.exists(save_results):
        os.makedirs(save_results)

    pipe = StableDiffusionXLPipeline.from_pretrained(sdxl_ckpt, use_safetensors=True)
    pipe.to('cpu')
    pipe.scheduler = TCDScheduler.from_config(pipe.scheduler.config)
    pipe.load_lora_weights(lora_weights)
    pipe.fuse_lora()

    hmunet = HmUnet(pipe.unet, model_path)
    pipe.unet = hmunet

    if prompt is not None:
        prompts = [prompt for _ in range(test_num)]
        negative_prompts = [[neg_prompt] for _ in range(test_num)]
    else:
        prompts = [str() for _ in range(test_num)]
        negative_prompts = [[str()] for _ in range(test_num)]

        prompts_list = [
            # 一位穿着粉色连衣裙的可爱年轻女孩在花园里微笑。照片级真实感，8K。
            'A cute young girl in a pink dress, smiling in a garden. Photo - realistic, 8k.',
            # 山间的瀑布，绿树和清澈的水。照片级真实，8K。
            'A waterfall in the mountains, green trees and clear water. Photo - real, 8k.',
            # 一只威严的龙在神奇森林中的场景，包含龙的特征、环境元素等。
            'A majestic dragon with iridescent scales, spreading its wings in a magical forest, glowing mushrooms on the ground, fire in its mouth, highly detailed.',
        ]

        negative_prompts_list = [
            # 避免生成模糊、低质量、变形、颜色难看、构图不佳以及卡通化的图像。
            [
                'Blurry, low - quality, distorted, ugly colors, bad composition, cartoonish.'
            ],
            # 避免生成低质量，虚假质感的水。
            ['Poor details, fake - looking water.'],
            # 防止生成单色、扁平、背景单调以及解剖结构错误的奇幻生物图像。
            ['Monochrome, flat, boring background, bad anatomy.'],
        ]

        for i in range(test_num):
            prompts[i] = prompts_list[i % 3]
            negative_prompts[i] = negative_prompts_list[i % 3]

    for i in range(test_num):
        start = time.time()
        image = pipe(
            prompt=prompts[i],
            negative_prompt=negative_prompts[i],
            guidance_scale=1,
            num_inference_steps=num_inference_steps,
            eta=eta,
        ).images[0]
        cost = time.time() - start
        save_path = os.path.join(save_results, f'{i}.jpg')
        image.save(save_path)
        unet_time = hmunet.profile["unet_infer"]
        avg_unet_time = unet_time / num_inference_steps
        hmunet.profile["unet_infer"] = 0
        logger.success(f"demo results saved in {save_path}, cost {cost*1000:.3f} ms.")
        logger.success(
            f"unet infer {num_inference_steps} steps cost {unet_time*1000:.3f} ms, average {avg_unet_time*1000:.3f} ms."
        )
