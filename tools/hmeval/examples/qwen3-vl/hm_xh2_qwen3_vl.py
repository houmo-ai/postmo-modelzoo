import os
import pkg_resources
from pathlib import Path
from typing import List, Optional, Dict, Any
from hm_xh2_qwen3_vl_impl import Qwen3VL
from evalscope.api.model import ModelAPI, GenerateConfig, ModelOutput
from evalscope.api.messages import ChatMessage
from evalscope.api.tool import ToolChoice, ToolInfo
from evalscope.api.registry import register_model_api
from evalscope.api.tool.utils import logger

API_NAME = "hm_xh2_qwen3_vl"


def has_tail_loop(text: str, sub_str_len=32, repeat=5):
    if len(text) < sub_str_len:
        return False
    sub_str = text[-sub_str_len:]
    if text.count(sub_str) > repeat:
        return True
    return False


@register_model_api(name=API_NAME)
class HmXH2Qwen3VL(ModelAPI):
    """Custom Qwen3-VL model implementation."""

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

        # Check transformers version
        self._check_transformers_version()

        model_path = model_args.get("model_dir")
        if not model_path:
            raise ValueError("`model_dir` is required in model_args.")
        if not os.path.isdir(model_path):
            raise ValueError(f"Model directory does not exist: {model_path}")

        tokenizer_dir = model_args.get("tokenizer_dir")
        if not tokenizer_dir:
            raise ValueError("`tokenizer_dir` is required in model_args.")
        if not os.path.isdir(tokenizer_dir):
            raise ValueError(f"Tokenizer directory does not exist: {tokenizer_dir}")

        normalized_model_dir = os.path.normpath(model_path)
        model_basename = os.path.basename(normalized_model_dir)
        self.model_name = model_basename if model_basename else "custom_model"

        keywords = ["prefill", "decode", "visual"]
        hmm_files = self.find_hmm_files(model_path, keywords)

        missing_keys = [k for k, v in hmm_files.items() if v is None]
        if missing_keys:
            raise ValueError(
                f"Missing required hmm files for keywords: {missing_keys}, model_dir={model_path}"
            )

        hmm_prefill_path = os.path.join(model_path, hmm_files["prefill"])
        hmm_decode_path = os.path.join(model_path, hmm_files["decode"])
        hmm_visual_path = os.path.join(model_path, hmm_files["visual"])
        embedding_path = os.path.join(model_path, "hmquant", "quant_embedding.pt")
        if not os.path.exists(embedding_path):
            raise ValueError(f"Embedding path does not exist: {embedding_path}")

        logger.info(hmm_visual_path)
        logger.info(hmm_prefill_path)
        logger.info(hmm_decode_path)
        logger.info(embedding_path)

        self.model = Qwen3VL(
            vit_path=hmm_visual_path,
            prefill_path=hmm_prefill_path,
            decode_path=hmm_decode_path,
            embedding_path=embedding_path,
            tokenizer_dir=tokenizer_dir,
            temperature=1.0,
            topk=None,
            topp=1.0,
            repetition_penalty=0.0,
        )

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

    def _process_messages(self, messages: List[ChatMessage]) -> Dict:
        texts, images = list(), list()
        for message in messages:
            msg_id = getattr(message, "id", "unknown")
            contents = getattr(message, "content", list())
            if not contents:
                logger.warning(f"Message id: {msg_id} has no content.")
            for content in contents:
                content_type = getattr(content, "type")
                if content_type not in ["text", "image"]:
                    logger.warning(
                        f"Message id: {msg_id} has invalid content type: {content_type}"
                    )
                    continue
                if content_type == "text":
                    text = getattr(content, "text")
                    if text is None:
                        continue
                    texts.append(text)
                elif content_type == "image":
                    image: str = getattr(content, "image")
                    if image is None:
                        continue
                    images.append(image)

        return {"text": texts, "images": images}

    def _call_model(self, input_text: Dict, config: GenerateConfig) -> str:
        """Run local model inference and return decoded response text."""
        texts = input_text["text"]
        if len(texts) == 0:
            return ""
        text = texts[0]
        images = input_text["images"]
        image = None if len(images) == 0 else images[0]
        input_tokens = self.model.chat_vit_prefill(text, image)
        decode_count = 0
        while True:
            next_str = self.model.chat_decoder()
            # avoid potential infinite loop in generation
            if len(self.model.all_response) > 800 and has_tail_loop(
                self.model.all_response, sub_str_len=32, repeat=5
            ):
                logger.warning("Detected tail loop in generated text.")
                break
            decode_count += 1
            if next_str is None:
                break
        return f"Response to: {self.model.all_response}"

    def _check_transformers_version(self):
        """Check transformers version and fail fast instead of mutating the runtime environment."""
        required_version = "4.57.1"
        try:
            installed_version = pkg_resources.get_distribution("transformers").version
            if installed_version != required_version:
                raise RuntimeError(
                    "Transformers version mismatch for this example model: "
                    f"required={required_version}, installed={installed_version}. "
                    "Please switch to a compatible environment or install the required version manually."
                )
        except pkg_resources.DistributionNotFound:
            raise RuntimeError(
                "transformers package is not installed in current environment. "
                f"This example requires transformers=={required_version}."
            )

    @staticmethod
    def find_hmm_files(model_dir: str, keywords: List[str]) -> Dict[str, Optional[str]]:
        """
        Find the first file matching each keyword in the given directory.

        Args:
            model_dir (str): The directory to search in.
            keywords (List[str]): A list of keywords to search for in file names.

        Returns:
            Dict[str, Optional[str]]: A dictionary mapping each keyword to the first matching file name,
                                      or None if no match is found.
        """
        hmm_files = {}
        for keyword in keywords:
            matching_files = [
                p.name
                for p in Path(model_dir).rglob("*")
                if p.is_file() and keyword in p.name
            ]
            hmm_files[keyword] = matching_files[0] if matching_files else None
        return hmm_files
