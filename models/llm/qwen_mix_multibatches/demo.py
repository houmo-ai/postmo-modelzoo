import math
import time
import os

import numpy as np
import tcim
import torch
import torch.nn.functional as F
from loguru import logger
from transformers import AutoTokenizer

TOKENIZER_PATH = 'qwen1.5-7b-chat-hf'
MODEL_PATH = os.path.join('output', os.getenv('HOUMO_TARGET', ''), 'result')


def run_decoder_model(model, model2, input_data, valid_length):
    input_name0 = 'input_1'
    input_name1 = 'valid_length'
    input_name2 = 'current_length'
    # input_data0 = np.zeros([1, 1, 4096]).astype('int16')
    # input_data1 = np.array([64]).astype('int16')
    # input_data2 = np.array([1]).astype('int16')
    input_data0 = input_data
    input_data1 = np.array([i - 1 for i in valid_length]).astype('int16')
    input_data2 = np.array([1, 1, 1, 1]).astype('int16')
    model.set_input(input_name0, input_data0)
    model.set_input(input_name1, input_data1)
    model.set_input(input_name2, input_data2)
    decode_model1_output_addr = model.get_dev_output('model_layers_15_resadd2')
    model2.set_input('model_layers_15_resadd2', decode_model1_output_addr)
    model2.set_input('valid_length', model.get_dev_input('valid_length'))
    model2.set_input('current_length', model.get_dev_input('current_length'))
    model.run()
    model2.run()
    output = model2.get_output('model_layers_31_resadd2', True)
    output_eval = output.reshape([4, 1, 4096])
    out_shape = [1, 4, 1, 4096]
    output_eval = output_eval.reshape(out_shape)

    return torch.empty((0), dtype=torch.int)


def run_prefill_model(model, model2, input_data, current_length, valid_length=0):
    input_name0 = 'input_1'
    input_name1 = 'valid_length'
    input_name2 = 'current_length'
    input_data0 = input_data
    input_data1 = np.array([valid_length]).astype('int16')
    input_data2 = np.array([current_length]).astype('int16')
    model.set_input(input_name0, input_data0)
    model.set_input(input_name1, input_data1)
    model.set_input(input_name2, input_data2)
    prefill_model1_output_addr = model.get_dev_output(
        'model_layers_15_resadd2',
    )

    model2.set_input(input_name1, input_data1)
    model2.set_input(input_name2, input_data2)
    model2.set_input('model_layers_15_resadd2', prefill_model1_output_addr)
    model.run()
    model2.run()
    model2.sync()


def run_model_decoder_head(model, input_data):
    model.run()
    output = model.get_output('lm_head_add_list_0', True)

    return output


def run_model_prefill_head(model1, input_data, gather_index):
    input1_name = 'current_length'
    input1_data = np.array([gather_index]).astype('int16')
    model1.set_input(input1_name, input1_data)
    model1.run()
    model1.sync()
    output = model1.get_output('lm_head_add_list_0', True)

    return output


class HmQwen:
    def __init__(self):
        xx = [i for i in range(3, 131)]
        weight_manager = tcim.runtime.create_weight_manager()
        self.prefill_part2_model_batch1 = tcim.runtime.load(
            'qwen_prefill_part2.hmm', weight_manager=weight_manager,
        )
        self.prefill_part1_model_batch1 = tcim.runtime.load(
            'qwen_prefill_part1.hmm', weight_manager=weight_manager,
        )
        self.prefill_part2_model_batch2 = tcim.runtime.load(
            'qwen_prefill_part2.hmm', weight_manager=weight_manager,
        )
        self.prefill_part1_model_batch2 = tcim.runtime.load(
            'qwen_prefill_part1.hmm', weight_manager=weight_manager,
        )
        self.prefill_part2_model_batch3 = tcim.runtime.load(
            'qwen_prefill_part2.hmm', weight_manager=weight_manager,
        )
        self.prefill_part1_model_batch3 = tcim.runtime.load(
            'qwen_prefill_part1.hmm', weight_manager=weight_manager,
        )
        self.prefill_part2_model_batch4 = tcim.runtime.load(
            'qwen_prefill_part2.hmm', weight_manager=weight_manager,
        )
        self.prefill_part1_model_batch4 = tcim.runtime.load(
            'qwen_prefill_part1.hmm', weight_manager=weight_manager,
        )

        self.prefill_head_model = tcim.runtime.load(
            'qwen_prefill_head.hmm', weight_manager=weight_manager,
        )
        self.decoder_part1_model = tcim.runtime.load(
            'qwen_decode_part1.hmm', weight_manager=weight_manager, reuse_inputs=xx,
        )
        self.decoder_part2_model = tcim.runtime.load(
            'qwen_decode_part2.hmm', weight_manager=weight_manager, reuse_inputs=xx,
        )
        self.decoder_head_model = tcim.runtime.load(
            'qwen_decode_head.hmm', weight_manager=weight_manager,
        )
        self.qwen1_5tokenizer = AutoTokenizer.from_pretrained(
            f'{TOKENIZER_PATH}', trust_remote_code=True,
        )
        self.embedding_weight = torch.load(
            f'{MODEL_PATH}/qwen15_quant_embedding.pt', map_location='cpu',  # weights_only=True,
        ).reshape(-1, 4096)
        self.batch = 4
        self.stream = tcim.runtime.Stream()
        self.prefill_models = [
            self.prefill_part1_model_batch1, self.prefill_part2_model_batch1, self.prefill_part1_model_batch2, self.prefill_part2_model_batch2,
            self.prefill_part1_model_batch3, self.prefill_part2_model_batch3, self.prefill_part1_model_batch4, self.prefill_part2_model_batch4,
        ]

    def get_prefill_input_dev(self, model1, model2):
        batch1_kv_input_1 = []
        for i in range(32):
            if i % 2 == 0:
                batch1_kv_input_1.append(
                    model1.get_dev_input(
                        f'model_layers_{i // 2}_self_attn_kcache_input',
                    ),
                )
            else:
                batch1_kv_input_1.append(
                    model1.get_dev_input(
                        f'model_layers_{i // 2}_self_attn_vcache_input',
                    ),
                )
        for i in range(32, 64):
            if i % 2 == 0:
                batch1_kv_input_1.append(
                    model2.get_dev_input(
                        f'model_layers_{i // 2}_self_attn_kcache_input',
                    ),
                )
            else:
                batch1_kv_input_1.append(
                    model2.get_dev_input(
                        f'model_layers_{i // 2}_self_attn_vcache_input',
                    ),
                )
        return batch1_kv_input_1

    def set_decode_kv_input(self, prefill1_input, prefill2_input, prefill3_input, prefill4_input):
        prefill_all = [
            prefill1_input, prefill2_input,
            prefill3_input, prefill4_input,
        ]
        for batch in range(self.batch):
            prefill = prefill_all[batch]
            for i in range(16):
                self.decoder_part1_model.set_input(
                    f'model_layers_{i}_self_attn_kcache_input_batch{batch}', prefill[i * 2],
                )
                self.decoder_part1_model.set_input(
                    f'model_layers_{i}_self_attn_vcache_input_batch{batch}', prefill[i * 2 + 1],
                )
                self.decoder_part2_model.set_input(
                    f'model_layers_{i + 16}_self_attn_kcache_input_batch{batch}', prefill[i * 2 + 32],
                )
                self.decoder_part2_model.set_input(
                    f'model_layers_{i + 16}_self_attn_vcache_input_batch{batch}', prefill[32 + i * 2 + 1],
                )

    def chat(self, messages, idx=0, prefill_length=256):
        if len(messages) != self.batch:
            print('question 4batch please!')
            return

        start_time = time.time()
        batch1_kv_input = self.get_prefill_input_dev(
            self.prefill_part1_model_batch1, self.prefill_part2_model_batch1,
        )
        batch2_kv_input = self.get_prefill_input_dev(
            self.prefill_part1_model_batch2, self.prefill_part2_model_batch2,
        )
        batch3_kv_input = self.get_prefill_input_dev(
            self.prefill_part1_model_batch3, self.prefill_part2_model_batch3,
        )
        batch4_kv_input = self.get_prefill_input_dev(
            self.prefill_part1_model_batch4, self.prefill_part2_model_batch4,
        )
        self.set_decode_kv_input(
            batch1_kv_input, batch2_kv_input, batch3_kv_input, batch4_kv_input,
        )

        text_4 = [
            self.qwen1_5tokenizer.apply_chat_template(
                messages[i],
                tokenize=False,
                add_generation_prompt=True,
            ) for i in range(self.batch)
        ]
        inputs = [
            self.qwen1_5tokenizer(
                text_4[i], return_tensors='pt',
            ) for i in range(self.batch)
        ]
        all_input_ids = [inputs[i]['input_ids'] for i in range(self.batch)]
        input_echo_len = [all_input_ids[i].numel() for i in range(self.batch)]
        for i in range(self.batch):
            if input_echo_len[i] >= 2048:
                logger.error(
                    f'batch {i} Question too long, please shorten it!',
                )
                return 'Question too long, please shorten it!'

        all_next_id = []
        all_prefill_response = []
        for i in range(self.batch):
            curr_input_echo_len = input_echo_len[i]
            prefill_loop_round = math.ceil(curr_input_echo_len / 256)
            for round in range(prefill_loop_round):
                valid_length = round * 256
                if round == prefill_loop_round - 1:
                    current_length = curr_input_echo_len - round * 256
                    gather_index = current_length - 1
                    input_ids = all_input_ids[i][
                        :,
                        round * 256: curr_input_echo_len
                    ]
                else:
                    current_length = 256
                    input_ids = all_input_ids[i][
                        :,
                        round * 256: (round + 1) * 256
                    ]
                inputs_embeds = F.embedding(input_ids, self.embedding_weight)
                effective_length = input_ids.size(-1)
                _pad_embeds = torch.zeros(
                    1, prefill_length -
                    effective_length, inputs_embeds.size(-1),
                    dtype=inputs_embeds.dtype, device=inputs_embeds.device,
                )
                input_data = torch.cat([inputs_embeds, _pad_embeds], dim=1).reshape(
                    4, 64, 4096,
                )  # [256, 1, 4096] ==> [4, 64, 4096]
                run_prefill_model(
                    self.prefill_models[i * 2], self.prefill_models[
                        i *
                        2 + 1
                    ], input_data, current_length, valid_length,
                )
            prefill_output_addr = self.prefill_models[
                i *
                2 + 1
            ].get_dev_output('model_layers_31_resadd2')
            self.prefill_head_model.set_input(
                'model_layers_31_resadd2', prefill_output_addr,
            )
            input_data = run_model_prefill_head(
                self.prefill_head_model, input_data, gather_index,
            )

            next_id = input_data.argmax(-1)
            all_next_id.append(next_id)
            prefill_response = self.qwen1_5tokenizer.decode(next_id.tolist())
            all_prefill_response.append(prefill_response)
        prefill_time = time.time() - start_time

        next_ids = [torch.from_numpy(i) for i in all_next_id]
        input_data = [
            F.embedding(next_id.unsqueeze(0), self.embedding_weight).reshape(
                1, 1, -1,
            ) for next_id in next_ids
        ]
        decode_input = torch.cat(input_data, dim=0)
        context_length = input_echo_len
        decode_count = [0] * 4
        decode_break = [0] * 4
        decode_loop = 0
        start_time = time.time()
        while True:
            if max(context_length) > 2048:
                logger.info(f'context length greater than 2048, break!')
                break
            decode_data = run_decoder_model(
                self.decoder_part1_model, self.decoder_part2_model, decode_input, context_length,
            )
            decode_output_addr = self.decoder_part2_model.get_dev_output(
                'model_layers_31_resadd2',
            )
            self.decoder_head_model.set_input(
                'reshape', decode_output_addr,
            )
            decode_head_data = run_model_decoder_head(
                self.decoder_head_model, decode_data,
            )
            next_ids = [
                decode_head_data[i].argmax(-1) for i in range(self.batch)
            ]
            decode_response = [
                self.qwen1_5tokenizer.decode(
                    next_ids[i],
                ) for i in range(self.batch)
            ]
            decode_4batch_res = []
            for batch in range(self.batch):
                if decode_response[batch] == self.qwen1_5tokenizer.eos_token:
                    decode_break[batch] = 1
                else:
                    context_length[batch] = context_length[batch] + 1
                    decode_count[batch] += 1
                next_id = torch.from_numpy(np.array(next_ids[batch]))
                input_data_i = F.embedding(
                    next_id.unsqueeze(
                        0,
                    ), self.embedding_weight,
                ).reshape(1, 1, -1)
                decode_4batch_res.append(input_data_i)
                if (decode_break[batch] == 0):
                    all_prefill_response[batch] = all_prefill_response[batch] + \
                        decode_response[batch]
            if sum(decode_break) == 4:
                break
            decode_input = torch.cat(decode_4batch_res, dim=0)
            decode_loop += 1
        decode_time = time.time() - start_time
        return all_prefill_response, decode_count, prefill_time, decode_time


if __name__ == '__main__':
    hmqwen = HmQwen()
    questions = [
        '请介绍一下存算一体技术的优势',
        '你是谁',
        "1+1=?",
        'Introduce yourself',
    ]
    logger.success('question:')
    print(f'\n{questions}')
    messages = []
    for i in range(4):
        messages.append([
            {'role': 'system', 'content': 'You are a helpful assistant.'},
            {'role': 'user', 'content': questions[i]},
        ])

    start_time = time.time()
    response, tokens, prefill_time, decode_time = hmqwen.chat(messages)
    total_time = time.time() - start_time

    for i in range(4):
        print(f'\nresponse{i}: {response[i]}')

    logger.success(f'total: {max(tokens)} tokens, cost {total_time:.3f} s')
    logger.success(
        f'prefill time: {prefill_time * 1000:.3f} ms, {1 / prefill_time:.2f} tokens/s',
    )
    decode_latency = decode_time * 1000 / (max(tokens) - 1)
    logger.success(
        f'decode average time: {decode_latency:.3f} ms, {1000 / decode_latency:.2f} tokens/s',
    )
    res_latency = total_time * 1000 / max(tokens)
    logger.success(
        f'end2end average time: {res_latency:.3f} ms, {1000 / res_latency:.2f} tokens/s',
    )

