# Copyright 2025 HOUMO AI
#
# File: gemma4_moe_mtp.py
# Description:
#   Gemma4-MoE MTP (Multi-Token Prediction) Speculative Decoding for HMM Inference.
#   Text-only - no vision or audio.
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

# fmt: off
import os
import sys
import math
import time
import numpy as np
import torch
from transformers import GemmaTokenizer, Gemma4Processor
from loguru import logger
import tcim_lite as tcim

PATCH_SIZE = 16


def is_valid_char(cp):
    return (0x4E00 <= cp <= 0x9FFF or 0x3400 <= cp <= 0x4DBF
            or 0x20000 <= cp <= 0x2A6DF or 0x0041 <= cp <= 0x005A or 0x0061 <= cp <= 0x007A)


class Gemma4MoEMTP:
    """Gemma4-MoE MTP speculative decoding (HMM backend, text-only).

    MTP speculative decoding flow (no target decode in loop):
      1. Target prefill -> first token + hidden state
      2. Draft model -> propose N draft tokens (reads target KV caches)
      3. Verify model -> batch-verify N+1 tokens (writes KV cache for all
         positions; compare logits with draft tokens to determine acceptance)
      4. At rejection position k, use verify logits[k] as the next token;
         accepted positions 0..k-1 plus verify prediction at k are committed
      5. Repeat from step 2 (draft model overwrites rejected KV cache slots)

    Model files (compiled .hmm for tcim backend):
      - prefill_path: target prefill model
      - decode_path: target decode model (not used in MTP loop)
      - verify_path: batch verify model (is_prefill=True, verify_length=N+1)
      - mtp_draft_path: assistant draft model, takes [token_emb || hidden_state]
          and shares KV caches with target
    """

    @staticmethod
    def _log_model_io(model, model_name):
        """Log input/output info for a model, marking KV-cache inputs."""
        n_in = model.get_num_inputs()
        n_out = model.get_num_outputs()
        logger.info(f"[{model_name}] {n_in} inputs, {n_out} outputs:")
        for i in range(n_in):
            name = model.get_input_name(i)
            if "cache" in name.lower():
                continue
            info = model.get_input_info(name)
            logger.info(f"  in[{i}]: {name} shape={info.shape} dtype={np.dtype(info.dtype).name}")
        for i in range(n_out):
            name = model.get_output_name(i)
            info = model.get_output_info(name)
            logger.info(f"  out[{i}]: {name} shape={info.shape} dtype={np.dtype(info.dtype).name}")

    def __init__(
        self,
        prefill_path,
        verify_path,
        mtp_draft_path,
        embedding_path=None,
        tokenizer_dir=None,
        devices=0,
        max_new_tokens=2048,
        enable_thinking=False,
    ):
        self.enable_thinking = enable_thinking
        self.max_new_tokens = max_new_tokens
        self.image_token_id = 258880
        self.pad_token_id = 0
        self.sliding_window = 1024

        if isinstance(devices, int):
            devices = [devices]
        self.devices = devices

        backend_name = "Xh2HalBackend"
        self.tokenizer = GemmaTokenizer.from_pretrained(tokenizer_dir, trust_remote_code=True)
        self.processor = Gemma4Processor.from_pretrained(tokenizer_dir, trust_remote_code=True)

        # ---- Load target prefill (allocates KV caches) ----
        dm = tcim.runtime.DevManager(self.devices, backend_name)
        wm = tcim.runtime.WeightManager(dm)

        logger.info(f"Loading prefill model from {prefill_path}")
        self.prefill = tcim.runtime.load(prefill_path, option=tcim.runtime.Option(wm))
        self._log_model_io(self.prefill, "prefill")
        self.prefill_len = self.prefill.get_input_info(self.prefill.get_input_name(0)).shape[1]
        self.embed_dim = self.prefill.get_input_info(self.prefill.get_input_name(0)).shape[2]
        self.global_mask_w = self.prefill.get_input_info(self.prefill.get_input_name(4)).shape[3]
        self.context_max_length = self.global_mask_w
        logger.info(f"Prefill loaded: len={self.prefill_len}, embed_dim={self.embed_dim}")

        cache_names = [
            self.prefill.get_input_name(i)
            for i in range(self.prefill.get_num_inputs())
            if "cache" in self.prefill.get_input_name(i).lower()
        ]
        opt = tcim.runtime.Option(wm)
        if cache_names:
            opt.set_dummy_tensors(cache_names)

        logger.info(f"Loading verify model from {verify_path}")
        self.verify = tcim.runtime.load(verify_path, option=opt)
        self._log_model_io(self.verify, "verify")
        for name in cache_names:
            self.verify.set_input(name, self.prefill.get_dev_input(name))

        self.verify_len = self.verify.get_input_info(self.verify.get_input_name(0)).shape[1]
        self.verify_embed_dim = self.verify.get_input_info(self.verify.get_input_name(0)).shape[2]
        self.num_draft_tokens = self.verify_len - 1
        logger.info(f"Verify loaded: len={self.verify_len}, embed_dim={self.verify_embed_dim}")
        
        draft_dummy_tensors = [
            "shared_key_cache_sliding", 
            "shared_value_cache_sliding",
            "shared_key_cache_full", 
            "shared_value_cache_full", 
        ]
        last_cache_names = self.get_last_cache_names()
        draft_opt = tcim.runtime.Option(wm)
        draft_opt.set_dummy_tensors(draft_dummy_tensors)
        logger.info(f"Loading MTP draft model from {mtp_draft_path}")
        self.draft = tcim.runtime.load(mtp_draft_path, option=draft_opt)
        self._log_model_io(self.draft, "mtp_draft")
        for idx, name in enumerate(draft_dummy_tensors):
            logger.info(f"Bind {last_cache_names[idx]} ==> {name}")
            self.draft.set_input(name, self.prefill.get_dev_input(last_cache_names[idx]))
        
        # ---- Embedding (index-based) ----
        emb = torch.load(embedding_path, map_location="cpu", weights_only=True)
        self.embedding = emb["weight"] if isinstance(emb, dict) else emb
        self.embedding = self.embedding.reshape(-1, self.embed_dim).float()

    def get_last_cache_names(self):
        kcache_names = [self.prefill.get_input_name(i) for i in range(self.prefill.get_num_inputs()) if "key_cache" in self.prefill.get_input_name(i).lower()]
        vcache_names = [self.prefill.get_input_name(i) for i in range(self.prefill.get_num_inputs()) if "value_cache" in self.prefill.get_input_name(i).lower()]
        last_kcache_local, last_kcache_global = kcache_names[-2], kcache_names[-1]
        last_vcache_local, last_vcache_global = vcache_names[-2], vcache_names[-1]
        return [last_kcache_local, last_vcache_local, last_kcache_global, last_vcache_global]
    
    @staticmethod
    def _aligned(size: int, align: int) -> int:
        return ((size + align - 1) // align) * align

    def _build_masks(self, cur_len, past_len, mask_len=None):
        q_len = mask_len
        neg = torch.tensor(torch.finfo(torch.float16).min, dtype=torch.float16)
        global_ctx = self.context_max_length

        global_mask = torch.full((1, 1, q_len, global_ctx), neg, dtype=torch.float16)
        valid_k = min(global_ctx, max(1, past_len + cur_len))
        for q in range(q_len):
            if q < cur_len:
                global_mask[0, 0, q, :min(valid_k, past_len + q + 1)] = 0

        sw = self.sliding_window
        slide_ctx = global_ctx if sw is None else min(global_ctx, self._aligned(sw + q_len - 1, 16))
        clamped_past = min(past_len, sw - 1) if sw is not None and sw > 0 else past_len
        local_mask = torch.full((1, 1, q_len, slide_ctx), neg, dtype=torch.float16)
        for q in range(q_len):
            if q < cur_len:
                causal_end = min(slide_ctx, clamped_past + q + 1)
                sw_start = max(0, clamped_past + q - sw + 1) if sw is not None else 0
                local_mask[0, 0, q, sw_start:causal_end] = 0

        return global_mask.numpy(), local_mask.numpy()

    def _build_embeddings(self, input_ids):
        embeds = self.embedding[input_ids[0]].unsqueeze(0).to(torch.float16)
        return embeds

    def _prefill(self, embeds, input_len):
        steps = math.ceil(input_len / self.prefill_len)
        for s in range(steps):
            start, end = s * self.prefill_len, min((s + 1) * self.prefill_len, input_len)
            cur_len = end - start
            sub_emb = embeds[:, start:end]
            if sub_emb.shape[1] < self.prefill_len:
                sub_emb = torch.cat([sub_emb, torch.zeros(1, self.prefill_len - sub_emb.shape[1], sub_emb.shape[2])], dim=1)

            g_mask, l_mask = self._build_masks(cur_len, start, self.prefill_len)

            self.prefill.set_input(self.prefill.get_input_name(0), sub_emb.numpy().astype(np.float16))
            self.prefill.set_input(self.prefill.get_input_name(1), np.array([start], dtype="int32"))
            self.prefill.set_input(self.prefill.get_input_name(2), np.array([cur_len], dtype="int32"))
            self.prefill.set_input(self.prefill.get_input_name(3), l_mask.astype(np.float16))
            self.prefill.set_input(self.prefill.get_input_name(4), g_mask.astype(np.float16))
            self.prefill.run()
            self.prefill.sync()

        next_id = self.prefill.get_output(self.prefill.get_output_name(0)).numpy().argmax(-1)
        hidden = torch.from_numpy(self.prefill.get_output(self.prefill.get_output_name(1)).numpy()).to(torch.float16)
        return next_id, hidden

    def _batch_verify(self, verify_tokens, past_len):
        tok = torch.from_numpy(np.array([verify_tokens]))
        verify_emb = self.embedding[tok].to(torch.float16)
        cur_len = len(verify_tokens)

        if cur_len < self.verify_len:
            pad = torch.zeros(1, self.verify_len - cur_len, self.verify_embed_dim, dtype=torch.float16)
            verify_emb = torch.cat([verify_emb, pad], dim=1)

        g_mask, l_mask = self._build_masks(cur_len, past_len, self.verify_len)

        self.verify.set_input(self.verify.get_input_name(0), verify_emb.numpy().astype(np.float16))
        self.verify.set_input(self.verify.get_input_name(1), np.array([past_len], dtype="int32"))
        self.verify.set_input(self.verify.get_input_name(2), np.array([cur_len], dtype="int32"))
        self.verify.set_input(self.verify.get_input_name(3), l_mask.astype(np.float16))
        self.verify.set_input(self.verify.get_input_name(4), g_mask.astype(np.float16))
        self.verify.run()
        self.verify.sync()

        logits = self.verify.get_output(self.verify.get_output_name(0)).numpy().astype(np.float32)
        logits = logits[:, -cur_len:, :].squeeze(0)  # (cur_len, vocab_size)

        hidden = torch.from_numpy(self.verify.get_output(self.verify.get_output_name(1)).numpy()).to(torch.float16)
        if hidden.shape[1] > cur_len:
            hidden = hidden[:, -cur_len:, :]
        return logits, hidden

    def _mtp_draft_step(self, tok_id, hidden_state, position):
        tok = torch.from_numpy(np.array([[tok_id]]))
        tok_emb = self.embedding[tok].reshape(1, 1, -1).to(torch.float16)
        mtp_input = torch.cat([tok_emb, hidden_state.to(torch.float16)], dim=-1)  # [1, 1, 5632]

        g_mask, l_mask = self._build_masks(1, position, 1)

        self.draft.set_input(self.draft.get_input_name(0), mtp_input.numpy().astype(np.float16))
        self.draft.set_input(self.draft.get_input_name(1), np.array([position], dtype=np.int32))
        self.draft.set_input(self.draft.get_input_name(2), l_mask.astype(np.float16))
        self.draft.set_input(self.draft.get_input_name(3), g_mask.astype(np.float16))

        self.draft.run()
        self.draft.sync()

        draft_logits = self.draft.get_output(self.draft.get_output_name(0)).numpy().astype(np.float32)
        draft_token = int(np.argmax(draft_logits[0, -1, :]))

        hidden = torch.from_numpy(self.draft.get_output(self.draft.get_output_name(1)).numpy()).to(torch.float16)
        return draft_token, hidden

    # ---- Chat ----

    def chat(self, question=""):
        """Run MTP speculative decoding (no target decode in loop).

        Algorithm per round:
          1. Draft: assistant model proposes num_draft_tokens candidates
          2. Verify: batch verify model checks N+1 tokens, writes KV cache
          3. Accept: compare verify logits with draft tokens
             - Mismatch at position k: accept 0..k-1, use verify prediction at k
             - All match: accept all N, use verify prediction at position N
          4. Advance past_len; next round overwrites rejected KV cache slots
          5. Repeat from step 1
        """
        q_text = question or "你好，请介绍一下你自己。"
        logger.success(f"question: {q_text}")

        content = [{"type": "text", "text": q_text}]

        inputs = self.processor.apply_chat_template(
            [{"role": "user", "content": content}],
            add_generation_prompt=True, tokenize=True, return_dict=True,
            return_tensors="pt", enable_thinking=self.enable_thinking)

        input_ids = inputs["input_ids"]
        input_len = input_ids.shape[-1]
        logger.info(f"Input length: {input_len}")
        if input_len >= self.context_max_length:
            logger.error(f"Input too long: {input_len}")
            sys.exit(1)

        # ---- Prefill ----
        embeds = self._build_embeddings(input_ids)
        first_token_id, prefill_hidden = self._prefill(embeds, input_len)
        logger.info(f"Prefill done, first token: {self.tokenizer.decode([first_token_id[0]])}")

        eos_ids = {self.tokenizer.eos_token_id} if isinstance(self.tokenizer.eos_token_id, int) else set(self.tokenizer.eos_token_id)
        eos_ids.add(106)

        # ---- MTP decode loop ----
        output_ids = input_ids[0].tolist()
        past_len = input_len
        next_token = int(first_token_id[0][0])
        current_hidden = prefill_hidden

        mtp_stats = {"verify_rounds": 0, "draft_proposed": 0, "draft_accepted": 0}

        logger.success("response:")
        print(f"\033[1;95m{self.tokenizer.decode([next_token])}", end="", flush=True)
        output_ids.append(next_token)
        generated = 1

        if next_token in eos_ids or generated >= self.max_new_tokens:
            print(f"\033[0m")
            return

        t0 = time.time()

        while past_len < self.context_max_length and generated < self.max_new_tokens:
            # ---- Phase 1: Draft (N tokens) ----
            draft_tokens = []
            assistant_hidden = current_hidden
            assistant_last_token_id = next_token
            assistant_position = max(past_len, 0)

            for _ in range(self.num_draft_tokens):
                draft_token, new_hidden = self._mtp_draft_step(assistant_last_token_id, assistant_hidden, assistant_position)
                draft_tokens.append(draft_token)
                assistant_last_token_id = draft_token
                assistant_hidden = new_hidden

            if not draft_tokens:
                break

            # ---- Phase 2: Batch Verify ----
            verify_tokens = [next_token] + draft_tokens
            verify_logits, verify_hidden = self._batch_verify(verify_tokens, past_len)

            # ---- Phase 3: Accept ----
            accepted_count = 0
            for idx in range(len(draft_tokens)):
                target_pred = int(np.argmax(verify_logits[idx]))
                if target_pred != draft_tokens[idx]:
                    break
                accepted_count += 1

            mtp_stats["verify_rounds"] += 1
            mtp_stats["draft_proposed"] += len(draft_tokens)
            mtp_stats["draft_accepted"] += accepted_count

            # ---- Phase 4: Next token from verify logits ----
            next_token = int(np.argmax(verify_logits[accepted_count]))

            idx = min(accepted_count, verify_hidden.shape[1] - 1)
            current_hidden = verify_hidden[:, idx:idx+1, :]

            past_len += 1 + accepted_count

            # ---- Phase 5: Output ----
            for d_token in draft_tokens[:accepted_count]:
                output_ids.append(d_token)
                generated += 1
                resp = self.tokenizer.decode([d_token])
                if resp:
                    print(resp, end="", flush=True)
                if d_token in eos_ids or generated >= self.max_new_tokens:
                    print(f"\033[0m")
                    self._print_mtp_stats(mtp_stats, time.time() - t0)
                    return

            output_ids.append(next_token)
            generated += 1
            resp = self.tokenizer.decode([next_token])
            if resp:
                print(resp, end="", flush=True)

            if next_token in eos_ids or generated >= self.max_new_tokens:
                print(f"\033[0m")
                self._print_mtp_stats(mtp_stats, time.time() - t0)
                return

        print(f"\033[0m")
        elapsed = time.time() - t0
        logger.info(f"Decode: {generated} tokens in {elapsed:.2f}s")
        self._print_mtp_stats(mtp_stats, elapsed)

    @staticmethod
    def _print_mtp_stats(stats, elapsed):
        rounds = stats["verify_rounds"]
        proposed = stats["draft_proposed"]
        accepted = stats["draft_accepted"]
        if rounds > 0:
            avg_accepted = accepted / rounds
            accept_rate = accepted / proposed if proposed > 0 else 0.0
            logger.info(
                f"[MTP stats] rounds={rounds} proposed={proposed} accepted={accepted} "
                f"avg_accepted_per_round={avg_accepted:.3f} "
                f"accept_rate={accept_rate * 100:.2f}% "
                f"elapsed={elapsed:.2f}s")
            print(
                f"[MTP stats] rounds={rounds} num_draft_per_round={proposed // rounds} "
                f"avg_accepted_per_round={avg_accepted:.3f} "
                f"accept_rate={accept_rate * 100:.2f}% ({accepted}/{proposed}) "
                f"elapsed={elapsed:.2f}s")
        else:
            logger.info("[MTP stats] no verify rounds executed.")
