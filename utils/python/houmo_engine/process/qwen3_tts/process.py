# Copyright (c) 2026 HOUMO AI
#
# File: process.py
# Description:
#   Qwen3-TTS input and output Process implementation.
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

"""Qwen3-TTS CPU preprocessing, embedding lookups, and codec postprocessing.

The Process converts user text, language, and speaker choices into the CPU
tensors that the Talker and Code Predictor graphs consume. It also owns the
codec-frame buffer used by streaming and the final codec trimming used by
oneshot. All device graphs (text projection, talker, code predictor, and audio
decoders) belong to :class:`Qwen3TtsModule`; stage ordering and sampling belong
to :class:`Qwen3TtsEngine`.

Text hidden states must be projected through the ``text_projection`` graph
before they enter the Talker input space. Because that projection runs on the
device, the Process returns the raw (CPU) text embedding slices in
:meth:`preprocess`, the Engine projects them through the Module, and the Process
assembles the final Talker input in :meth:`build_talker_input`. The projection
slices mirror the original single-file demo exactly rather than assuming the
projection is position independent.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import torch
from torch import Tensor

from ...core import ModelProcess
from ...core.types import GenerationState, StageInputs
from ...perf import PerfTracker


@dataclass
class Qwen3TtsGenerationState(GenerationState):
    """TTS request state: generated codec frames instead of text tokens."""

    codec_frames: List[Tensor] = field(default_factory=list)
    trailing_text_hidden: Optional[Tensor] = None
    talker_input_length: int = 0
    talker_generated_ids: Optional[Tensor] = None
    talker_generated_len: int = 0
    finished: bool = False


@dataclass
class Qwen3TtsPreparedRequest:
    """CPU tensors describing one synthesis request before Talker prefill."""

    input_id: Tensor
    codec_input_embedding: Tensor
    non_streaming_mode: bool
    # Raw (unprojected) text embedding slices. The Engine runs them through the
    # text_projection graph and passes the results back to build_talker_input.
    role_text_embed: Tensor
    body_text_embed: Optional[Tensor] = None
    head_text_embed: Optional[Tensor] = None
    trailing_text_embed: Optional[Tensor] = None


def _to_torch(value) -> Tensor:
    if isinstance(value, Tensor):
        return value
    return torch.from_numpy(np.asarray(value))


def build_assistant_text(text: str) -> str:
    """Build assistant-formatted prompt text (mirrors the original demo)."""
    return f"<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n"


class Qwen3TtsProcess(ModelProcess):
    """Qwen3-TTS text/codec preprocessing and codec postprocessing."""

    def __init__(
        self,
        hf_model_dir,
        talker_token_embedding_path,
        talker_text_embedding_path,
        code_predictor_token_embedding_path,
        *,
        perf: PerfTracker,
    ):
        self.perf = perf
        self.hf_model_dir = str(hf_model_dir)
        self._load_config_and_processor(self.hf_model_dir)
        self._load_embeddings(
            talker_token_embedding_path,
            talker_text_embedding_path,
            code_predictor_token_embedding_path,
        )
        self._init_static_cpu_embeddings()
        # Projected special-token embeddings; filled by the Engine after it runs
        # the text_projection graph once during initialization.
        self.tts_bos_embed: Optional[Tensor] = None
        self.tts_eos_embed: Optional[Tensor] = None
        self.tts_pad_embed: Optional[Tensor] = None

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------
    def _load_config_and_processor(self, model_path: str) -> None:
        from transformers import AutoConfig, AutoProcessor
        from qwen_tts.core.models.modeling_qwen3_tts import Qwen3TTSConfig
        from qwen_tts.core.models.processing_qwen3_tts import Qwen3TTSProcessor

        AutoConfig.register("qwen3_tts", Qwen3TTSConfig)
        AutoProcessor.register(Qwen3TTSConfig, Qwen3TTSProcessor)

        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        self.config = config
        self.talker_config = config.talker_config
        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

        self.tts_bos_token_id = config.tts_bos_token_id
        self.tts_eos_token_id = config.tts_eos_token_id
        self.tts_pad_token_id = config.tts_pad_token_id

        self.codec_eos_token_id = self.talker_config.codec_eos_token_id
        self.codec_bos_id = self.talker_config.codec_bos_id
        self.codec_pad_id = self.talker_config.codec_pad_id
        self.codec_think_id = self.talker_config.codec_think_id
        self.codec_nothink_id = self.talker_config.codec_nothink_id
        self.codec_think_bos_id = self.talker_config.codec_think_bos_id
        self.codec_think_eos_id = self.talker_config.codec_think_eos_id
        self.codec_language_id = self.talker_config.codec_language_id
        self.spk_id = self.talker_config.spk_id
        self.spk_is_dialect = self.talker_config.spk_is_dialect
        self.num_code_groups = self.talker_config.num_code_groups
        self.vocab_size = self.talker_config.vocab_size

    def _load_embeddings(
        self,
        talker_token_embedding_path,
        talker_text_embedding_path,
        code_predictor_token_embedding_path,
    ) -> None:
        self.token_embedding = torch.nn.Embedding.from_pretrained(
            torch.load(str(talker_token_embedding_path), map_location="cpu")["weight"],
            freeze=True,
        ).to(torch.float16)
        self.text_embedding = torch.nn.Embedding.from_pretrained(
            torch.load(str(talker_text_embedding_path), map_location="cpu")["weight"],
            freeze=True,
        ).to(torch.float16)

        cp_state = torch.load(str(code_predictor_token_embedding_path), map_location="cpu")
        num_embeddings, embedding_dim = cp_state["0.weight"].shape
        cp_layers = [
            torch.nn.Embedding(num_embeddings=num_embeddings, embedding_dim=embedding_dim) for _ in range(len(cp_state))
        ]
        self.code_predictor_token_embedding = torch.nn.ModuleList(cp_layers)
        self.code_predictor_token_embedding.load_state_dict(cp_state)
        self.code_predictor_token_embedding.to(torch.float16)
        self.code_predictor_token_embedding.requires_grad_(False)

    def _init_static_cpu_embeddings(self) -> None:
        self.codec_pad_bos_embed = self.token_embedding(
            torch.tensor([[self.codec_pad_id, self.codec_bos_id]], dtype=torch.long)
        )
        self.codec_bos_embed = self.token_embedding(torch.tensor([[self.codec_bos_id]], dtype=torch.long))
        self._codec_pad_embed_cache: Dict[int, Tensor] = {}
        self._codec_prefill_embed_cache: Dict[Optional[int], Tensor] = {}

        self._speaker_embed_cache: Dict[str, Tensor] = {}
        for speaker_name, speaker_id in self.spk_id.items():
            self._speaker_embed_cache[speaker_name.lower()] = self.token_embedding(
                torch.tensor(speaker_id, dtype=torch.long)
            )

    # ------------------------------------------------------------------
    # Static special-token embeddings (projection driven by the Engine)
    # ------------------------------------------------------------------
    def special_text_embeds(self) -> Tensor:
        """Raw text embedding of [bos, eos, pad] awaiting projection."""
        ids = torch.tensor(
            [[self.tts_bos_token_id, self.tts_eos_token_id, self.tts_pad_token_id]],
            dtype=torch.long,
        )
        return self.text_embedding(ids)

    def set_special_embeds(self, projected) -> None:
        """Store projected [bos, eos, pad] embeddings for talker input building."""
        projected = _to_torch(projected)
        self.tts_bos_embed, self.tts_eos_embed, self.tts_pad_embed = projected.chunk(3, dim=1)

    # ------------------------------------------------------------------
    # Speaker / language resolution
    # ------------------------------------------------------------------
    def _get_speaker_embedding(self, speaker: str) -> Tensor:
        speaker_key = speaker.lower()
        if speaker_key not in self._speaker_embed_cache:
            raise NotImplementedError(f"Speaker {speaker} not implemented")
        return self._speaker_embed_cache[speaker_key]

    def _get_codec_pad_embedding(self, length: int) -> Tensor:
        if length not in self._codec_pad_embed_cache:
            self._codec_pad_embed_cache[length] = self.token_embedding(
                torch.tensor([[self.codec_pad_id] * length], dtype=torch.long)
            )
        return self._codec_pad_embed_cache[length]

    def _get_codec_prefill_embedding(self, language_id: Optional[int]) -> Tensor:
        if language_id in self._codec_prefill_embed_cache:
            return self._codec_prefill_embed_cache[language_id]
        if language_id is None:
            codec_prefill_list = [[self.codec_nothink_id, self.codec_think_bos_id, self.codec_think_eos_id]]
        else:
            codec_prefill_list = [
                [
                    self.codec_think_id,
                    self.codec_think_bos_id,
                    language_id,
                    self.codec_think_eos_id,
                ]
            ]
        self._codec_prefill_embed_cache[language_id] = self.token_embedding(
            torch.tensor(codec_prefill_list, dtype=torch.long)
        )
        return self._codec_prefill_embed_cache[language_id]

    def _get_language_id(self, language: str, speaker: str = None) -> Optional[int]:
        if language.lower() == "auto":
            language_id = None
        else:
            if language.lower() not in self.codec_language_id:
                raise NotImplementedError(f"Language {language} not implemented")
            language_id = self.codec_language_id[language.lower()]

        if (
            language.lower() in ["chinese", "auto"]
            and speaker is not None
            and speaker != ""
            and speaker.lower() in self.spk_is_dialect
            and self.spk_is_dialect[speaker.lower()]
        ):
            dialect = self.spk_is_dialect[speaker.lower()]
            language_id = self.codec_language_id[dialect]
        return language_id

    # ------------------------------------------------------------------
    # preprocess
    # ------------------------------------------------------------------
    def _tokenize(self, text: str) -> Tensor:
        input_text = build_assistant_text(text)
        input_data = self.processor(text=input_text, return_tensors="pt", padding=True)
        input_id = input_data["input_ids"]
        return input_id.unsqueeze(0) if input_id.dim() == 1 else input_id

    def preprocess(
        self,
        text: str,
        language: str,
        speaker: str,
        non_streaming_mode: bool = True,
    ) -> Qwen3TtsPreparedRequest:
        with self.perf.scope("tts.embedding.preprocess"):
            input_id = self._tokenize(text)

            speaker_embed = self._get_speaker_embedding(speaker)
            language_id = self._get_language_id(language, speaker)

            codec_input_embedding_0 = self._get_codec_prefill_embedding(language_id)
            codec_input_embedding_1 = self.codec_pad_bos_embed
            if speaker_embed is None:
                codec_input_embedding = torch.cat([codec_input_embedding_0, codec_input_embedding_1], dim=1)
            else:
                codec_input_embedding = torch.cat(
                    [
                        codec_input_embedding_0,
                        speaker_embed.view(1, 1, -1),
                        codec_input_embedding_1,
                    ],
                    dim=1,
                )

            role_text_embed = self.text_embedding(input_id[:, :3])
            request = Qwen3TtsPreparedRequest(
                input_id=input_id,
                codec_input_embedding=codec_input_embedding,
                non_streaming_mode=non_streaming_mode,
                role_text_embed=role_text_embed,
            )
            if non_streaming_mode:
                request.body_text_embed = self.text_embedding(input_id[:, 3:-5])
            else:
                request.head_text_embed = self.text_embedding(input_id[:, 3:4])
                request.trailing_text_embed = self.text_embedding(input_id[:, 4:-5])
        return request

    # ------------------------------------------------------------------
    # Talker input assembly (uses projected text hidden states)
    # ------------------------------------------------------------------
    def build_talker_input(
        self,
        request: Qwen3TtsPreparedRequest,
        role_hidden,
        *,
        body_hidden=None,
        head_hidden=None,
        trailing_hidden=None,
    ):
        """Assemble Talker input embeddings and trailing text hidden states."""
        role_hidden = _to_torch(role_hidden)
        codec_input_embedding = request.codec_input_embedding
        tts_bos_embed = self.tts_bos_embed
        tts_eos_embed = self.tts_eos_embed
        tts_pad_embed = self.tts_pad_embed

        _talker_input_embed = (
            torch.cat(
                (
                    tts_pad_embed.expand(-1, codec_input_embedding.shape[1] - 2, -1),
                    tts_bos_embed,
                ),
                dim=1,
            )
            + codec_input_embedding[:, :-1]
        )
        talker_input_embed = torch.cat((role_hidden, _talker_input_embed), dim=1)

        input_id = request.input_id
        if request.non_streaming_mode:
            body_hidden = _to_torch(body_hidden)
            talker_input_embed = talker_input_embed[:, :-1]
            talker_input_embed = torch.cat(
                [
                    talker_input_embed,
                    torch.cat((body_hidden, tts_eos_embed), dim=1)
                    + self._get_codec_pad_embedding(input_id[:, 3:-5].shape[1] + 1),
                    tts_pad_embed + self.codec_bos_embed,
                ],
                dim=1,
            )
            trailing_text_hidden = tts_pad_embed
        else:
            head_hidden = _to_torch(head_hidden)
            trailing_hidden = _to_torch(trailing_hidden)
            talker_input_embed = torch.cat(
                [
                    talker_input_embed,
                    head_hidden + codec_input_embedding[:, -1:],
                ],
                dim=1,
            )
            trailing_text_hidden = torch.cat((trailing_hidden, tts_eos_embed), dim=1)

        return talker_input_embed, trailing_text_hidden

    # ------------------------------------------------------------------
    # Per-frame CPU embedding helpers
    # ------------------------------------------------------------------
    def talker_token_embed(self, token: Tensor) -> Tensor:
        """CPU token embedding for a Talker codec-group-0 token."""
        embed = self.token_embedding(token)
        return embed.unsqueeze(1) if embed.dim() == 2 else embed

    def prepare_code_predictor_prefill(self, inputs_embeds, prefill_length: int):
        """Build padded Code Predictor prefill embeddings and chunk metadata.

        Returns (inputs_embeds, seq_length, prefill_loop_round) so the Engine can
        drive the chunked prefill loop. The KV cache reset is a device operation
        owned by the Module.
        """
        inputs_embeds = _to_torch(inputs_embeds)
        seq_length = inputs_embeds.shape[1]
        import math

        prefill_loop_round = math.ceil(seq_length / prefill_length)
        padding_len = prefill_loop_round * prefill_length - seq_length
        if padding_len > 0:
            embed_layer = self.code_predictor_token_embedding[0]
            padding_embeds = embed_layer(torch.zeros(1, padding_len, dtype=torch.long))
            inputs_embeds = torch.cat([inputs_embeds, padding_embeds], dim=1)
        return inputs_embeds, seq_length, prefill_loop_round

    def code_predictor_decode_embed(self, step: int, token: Tensor) -> Tensor:
        """CPU token embedding for a Code Predictor decode step."""
        embed = self.code_predictor_token_embedding[step](token)
        return embed.unsqueeze(1) if embed.dim() == 2 else embed

    def prepare_talker_decode_input(
        self,
        last_id_hidden: Tensor,
        predictor_embeds: List[Tensor],
        step: int,
        trailing_text_hidden: Tensor,
    ) -> Tensor:
        """Fuse the completed codec frame into the next Talker decode input."""
        codec_hiddens = torch.cat([last_id_hidden] + predictor_embeds, dim=1)
        inputs_embeds = codec_hiddens.sum(1, keepdim=True)
        if step < trailing_text_hidden.shape[1]:
            inputs_embeds = inputs_embeds + trailing_text_hidden[:, step].unsqueeze(1)
        else:
            inputs_embeds = inputs_embeds + self.tts_pad_embed
        return inputs_embeds

    # ------------------------------------------------------------------
    # Stage input construction
    # ------------------------------------------------------------------
    def pad_talker_prefill(self, talker_input_embed: Tensor, prefill_length: int):
        """Pad Talker input to a multiple of prefill_length; return (padded, seq_len)."""
        import math

        seq_length = talker_input_embed.shape[1]
        prefill_loop_round = math.ceil(seq_length / prefill_length)
        padding_len = prefill_loop_round * prefill_length - seq_length
        if padding_len > 0:
            padding_embeds = self.token_embedding(torch.zeros(1, padding_len, dtype=torch.long))
            talker_input_embed = torch.cat([talker_input_embed, padding_embeds], dim=1)
        return talker_input_embed, seq_length, prefill_loop_round

    def prepare_talker_prefill_chunk(self, chunk_embeds: Tensor, valid_length: int, current_length: int) -> StageInputs:
        return StageInputs(
            tensors=(
                chunk_embeds,
                np.array([valid_length], dtype=np.int32),
                np.array([current_length], dtype=np.int32),
            ),
            metadata={"current_length": current_length},
        )

    def prepare_talker_decode(self, inputs_embeds: Tensor, valid_length: int) -> StageInputs:
        return StageInputs(
            tensors=(inputs_embeds, np.array([valid_length], dtype=np.int32)),
            metadata={"valid_length": valid_length},
        )

    def prepare_code_predictor_prefill_chunk(
        self,
        chunk_embeds: Tensor,
        valid_length: int,
        current_length: int,
        generation_steps: int,
    ) -> StageInputs:
        return StageInputs(
            tensors=(
                chunk_embeds,
                np.array([valid_length], dtype=np.int32),
                np.array([current_length], dtype=np.int32),
                np.array([generation_steps], dtype=np.int32),
            ),
            metadata={"current_length": current_length},
        )

    def prepare_code_predictor_decode(
        self, inputs_embeds: Tensor, valid_length: int, generation_steps: int
    ) -> StageInputs:
        return StageInputs(
            tensors=(
                inputs_embeds,
                np.array([valid_length], dtype=np.int32),
                np.array([generation_steps], dtype=np.int32),
            ),
            metadata={"generation_steps": generation_steps},
        )

    # ------------------------------------------------------------------
    # Codec postprocessing (oneshot trimming)
    # ------------------------------------------------------------------
    def postprocess(self, state: Qwen3TtsGenerationState, *, final: bool = False):
        """Trim generated codec frames at the EOS boundary (oneshot path).

        Returns the list of valid codec-frame tensors ready for the speech
        tokenizer. Non-final calls return an empty list because oneshot does not
        stream partial audio.
        """
        if not final:
            return []
        with self.perf.scope("tts.postprocess"):
            generated_codes = state.codec_frames
            if not generated_codes:
                return []
            talker_codes = torch.stack(generated_codes, dim=1)
            first_codebook = talker_codes[:, :, 0]
            is_stop_token = first_codebook == self.codec_eos_token_id
            stop_indices = torch.argmax(is_stop_token.int(), dim=1)
            has_stop_token = is_stop_token.any(dim=1)
            effective_lengths = torch.where(has_stop_token, stop_indices, talker_codes.shape[1])
            return [talker_codes[i, :length] for i, length in enumerate(effective_lengths)]


__all__ = [
    "Qwen3TtsProcess",
    "Qwen3TtsPreparedRequest",
    "Qwen3TtsGenerationState",
    "build_assistant_text",
]
