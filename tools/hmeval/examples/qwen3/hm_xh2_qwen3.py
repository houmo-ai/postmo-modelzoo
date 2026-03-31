import os
import re
import math
import torch
import torch.nn.functional as F
import numpy as np
import tcim_lite as tcim
from transformers import Qwen2Tokenizer
from typing import List, Optional, Dict, Any
from evalscope.api.model import ModelAPI, GenerateConfig, ModelOutput
from evalscope.api.messages import ChatMessage
from evalscope.api.tool import ToolChoice, ToolInfo
from evalscope.api.registry import register_model_api
from evalscope.api.tool.utils import logger

API_NAME = "hm_xh2_qwen3"


@register_model_api(name=API_NAME)
class HmXH2Qwen3(ModelAPI):
    """Custom model implementation for hm_xh2 qwen3."""

    def __init__(
        self,
        model_name: str,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        config: GenerateConfig = GenerateConfig(),
        **model_args: Dict[str, Any],
    ) -> None:
        super().__init__(model_name, base_url, api_key, config)
        self.model_args = model_args
        logger.info(f"Model args: {self.model_args}")

        model_dir = model_args.get("model_dir")
        if not model_dir:
            raise ValueError("`model_dir` is required in model_args.")
        if not os.path.isdir(model_dir):
            raise ValueError(f"Model path {model_dir} does not exist.")

        normalized_model_dir = os.path.normpath(model_dir)
        model_basename = os.path.basename(normalized_model_dir)
        self.model_name = model_basename if model_basename else "custom_model"

        # Initialize model artifacts. Paths such as tokenizer_dir can be passed by --model-args.
        tokenizer_dir = model_args.get("tokenizer_dir")
        if tokenizer_dir is None or not os.path.isdir(tokenizer_dir):
            raise ValueError(f"Tokenizer path {tokenizer_dir} does not exist.")

        embedding_path = os.path.join(model_dir, "hmquant", "quant_embedding.pt")
        prefill_model_path = os.path.join(model_dir, "qwen3_prefill.hmm")
        if not os.path.exists(prefill_model_path):
            raise ValueError(f"Model path {prefill_model_path} does not exist.")
        decode_model_path = os.path.join(model_dir, "qwen3_decode.hmm")
        if not os.path.exists(decode_model_path):
            raise ValueError(f"Model path {decode_model_path} does not exist.")
        weight_manager = tcim.runtime.WeightManager(device=0)
        option1 = tcim.runtime.Option(weight_manager)
        option2 = tcim.runtime.Option(weight_manager)
        self.prefill = tcim.runtime.load(prefill_model_path, option1)
        logger.info("Loaded prefill model.")
        self.nblocks = self.get_nblocks()

        dummy_tensor_names = [
            f"model_layers_{i}_self_attn_kcache_input" for i in range(self.nblocks)
        ]
        dummy_tensor_names += [
            f"model_layers_{i}_self_attn_vcache_input" for i in range(self.nblocks)
        ]
        option2.set_dummy_tensors(dummy_tensor_names)
        self.decode = tcim.runtime.load(decode_model_path, option=option2)
        logger.info("Loaded decode model.")

        self.prefill_length = self.prefill.get_input_info(
            self.prefill.get_input_name(0)
        ).shape[1]
        self.embedding_len = self.prefill.get_input_info(
            self.prefill.get_input_name(0)
        ).shape[2]
        self.context_max_length = self.decode.get_input_info(
            self.decode.get_input_name(3)
        ).shape[2]

        # Initialize cache inputs for decode model from prefill model
        for i in range(3, 2 * self.nblocks + 3):
            cache = self.prefill.get_input(self.prefill.get_input_name(i))
            self.decode.set_input(self.decode.get_input_name(i), cache)

        # Set initial decode current length input
        current_length_input_1 = np.array([1]).astype("int32")
        decode_current_length_name = self.decode.get_input_name(2)
        self.decode.set_input(decode_current_length_name, current_length_input_1)

        # Load tokenizer and embedding weights
        self.tokenizer = Qwen2Tokenizer.from_pretrained(
            tokenizer_dir, trust_remote_code=True
        )
        embedding_weight = torch.load(embedding_path, map_location="cpu")
        embedding_weight = embedding_weight["weight"]
        self.embedding_weight = embedding_weight.reshape(-1, self.embedding_len).float()
        self.context_length = 0

    def get_nblocks(self) -> int:
        """Calculate number of transformer blocks from input tensor names."""
        input_names = []
        for i in range(self.prefill.get_num_inputs()):
            input_names.append(self.prefill.get_input_name(i))
        pattern = r"^model_layers_(\d+)_self_attn_kcache_input$"
        count = sum(1 for item in input_names if re.match(pattern, item))
        return count

    def generate(
        self,
        input: List[ChatMessage],
        tools: List[ToolInfo],
        tool_choice: ToolChoice,
        config: GenerateConfig,
    ) -> ModelOutput:
        # 1) Process input messages
        input_text = self._process_messages(input)

        # 2) Run model inference
        response = self._call_model(input_text, config)

        # 3) Return normalized output
        return ModelOutput.from_content(model=self.model_name, content=response)

    def _process_messages(self, messages: List[ChatMessage]) -> str:
        """Convert chat messages to plain text using the tokenizer template."""
        text_parts = [{"role": "system", "content": "You are a helpful assistant."}]
        for message in messages:
            role = getattr(message, "role", "user")
            content = getattr(message, "content", str(message))
            text_parts.append({"role": role, "content": content})
        return self.tokenizer.apply_chat_template(
            text_parts,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

    def _call_model(self, input_text: str, config: GenerateConfig) -> str:
        """Run local model inference and return decoded response text."""
        generated_ids = []
        self.context_length = 0
        inputs = self.tokenizer(
            input_text, return_tensors="pt", add_special_tokens=False
        )
        all_input_ids = inputs["input_ids"]
        input_echo_len = all_input_ids.numel()
        if input_echo_len >= self.context_max_length:
            logger.error(
                f"Input sequence length ({input_echo_len}) exceeds maximum context length ({self.context_max_length}), please shorten your question!"
            )
            return ""

        # prefill
        prefill_loop_round = math.ceil(input_echo_len / self.prefill_length)
        for idx in range(prefill_loop_round):
            valid_length = idx * self.prefill_length + self.context_length
            end_pos = (
                input_echo_len
                if idx == prefill_loop_round - 1
                else (idx + 1) * self.prefill_length
            )
            current_length = end_pos - idx * self.prefill_length
            input_ids = all_input_ids[:, idx * self.prefill_length : end_pos]
            inputs_embeds = F.embedding(input_ids, self.embedding_weight)
            effective_length = input_ids.size(-1)
            _pad_embeds = torch.zeros(
                1,
                self.prefill_length - effective_length,
                inputs_embeds.size(-1),
                dtype=inputs_embeds.dtype,
                device=inputs_embeds.device,
            )
            input_data = torch.cat([inputs_embeds, _pad_embeds], dim=1).reshape(
                1, self.prefill_length, self.embedding_len
            )

            valid_length_data = np.array([valid_length]).astype("int32")
            current_length_data = np.array([current_length]).astype("int32")

            input_name = self.prefill.get_input_name(0)
            valid_length_name = self.prefill.get_input_name(1)
            current_length_name = self.prefill.get_input_name(2)

            self.prefill.set_input(input_name, input_data.numpy())
            self.prefill.set_input(valid_length_name, valid_length_data)
            self.prefill.set_input(current_length_name, current_length_data)

            self.prefill.run()
            self.prefill.sync()

        input_data = self.prefill.get_output(self.prefill.get_output_name(0)).numpy()
        next_id = input_data.argmax(-1)[0]
        next_id = torch.from_numpy(next_id)
        self.context_length += input_echo_len
        generated_ids.append(next_id)

        while True:
            if self.context_length >= self.context_max_length:
                logger.info(
                    f"Context length ({self.context_length}) exceeds maximum limit ({self.context_max_length}), stopping generation!"
                )
                break
            input_data = F.embedding(
                next_id.unsqueeze(0), self.embedding_weight
            ).reshape(1, 1, -1)
            input_name = self.decode.get_input_name(0)
            valid_length_name = self.decode.get_input_name(1)
            self.decode.set_input(input_name, input_data.numpy())
            valid_length_data = np.array(self.context_length).astype("int32")
            self.decode.set_input(valid_length_name, valid_length_data)

            self.decode.run()
            self.decode.sync()
            input_data = self.decode.get_output(self.decode.get_output_name(0)).numpy()
            next_id = input_data.argmax(-1)[0]
            next_id = torch.from_numpy(next_id)

            if next_id == self.tokenizer.eos_token_id:
                break
            self.context_length += 1
            generated_ids.append(next_id)

        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        response = "".join(response)
        return f"Response to: {response}"
