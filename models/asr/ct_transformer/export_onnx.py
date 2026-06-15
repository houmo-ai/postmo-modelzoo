import argparse
from pathlib import Path

import onnx
import onnxruntime as ort
import torch
from torch import nn

from model_utils import freeze_encoder_position_encoder, load_auto_model

THIS_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_DIR = THIS_DIR / "ct_transformer"
DEFAULT_INPUT = DEFAULT_MODEL_DIR / "example/punc_example.txt"
DEFAULT_OUTPUT_DIR = THIS_DIR / "work_dirs" / "export_fp32"
DEFAULT_ONNX_PATH = DEFAULT_OUTPUT_DIR / "onnx" / "model.onnx"
DEFAULT_SIMPLIFIED_ONNX_PATH = DEFAULT_OUTPUT_DIR / "onnx" / "model_simplified.onnx"


def inspect_onnx(model_path: Path) -> None:
    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    print("Model inputs:")
    for inp in session.get_inputs():
        print(f"  {inp.name}: {inp.type}, shape={inp.shape}")
    print("Model outputs:")
    for out in session.get_outputs():
        print(f"  {out.name}: {out.type}, shape={out.shape}")

def load_text_items(input_path: Path) -> list[str]:
    items = []
    for line in input_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            items.append(line)
    if not items:
        raise ValueError(f"No valid text lines found in {input_path}")
    return items


def collect_max_token_length(auto_model, input_path: Path, split_size: int) -> int:
    from funasr.models.ct_transformer.utils import split_to_mini_sentence, split_words

    tokenizer = auto_model.kwargs["tokenizer"]
    jieba_usr_dict = auto_model.model.jieba_usr_dict

    max_len = 0
    for text in load_text_items(input_path):
        tokens = split_words(text, jieba_usr_dict=jieba_usr_dict)
        token_ids = tokenizer.encode(tokens)
        mini_sentences_id = split_to_mini_sentence(token_ids, split_size)
        cache_sent_id = []
        for mini_sentence_id in mini_sentences_id:
            merged = cache_sent_id + list(mini_sentence_id)
            max_len = max(max_len, len(merged))
            cache_sent_id = []
    if max_len <= 0:
        raise ValueError(f"Failed to derive token length from {input_path}")
    return max_len + 200  # Add buffer for sliding window cache without punctuation


class PuncForwardWrapper(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model.cpu().float().eval()

    def forward(self, text: torch.Tensor, text_lengths: torch.Tensor):
        logits, _ = self.model.punc_forward(text=text, text_lengths=text_lengths)
        return logits


def export_onnx(
    model_dir: Path,
    model_revision: str,
    input_path: Path,
    onnx_path: Path,
    opset_version: int,
    split_size: int,
) -> int:
    auto_model = load_auto_model(model_dir, model_revision)
    fixed_length = collect_max_token_length(auto_model, input_path, split_size)
    freeze_encoder_position_encoder(auto_model.model, fixed_length)
    export_model = PuncForwardWrapper(auto_model.model)
    dummy_text = torch.zeros((1, fixed_length), dtype=torch.int32)
    dummy_lengths = torch.tensor([fixed_length], dtype=torch.int32)

    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        export_model,
        (dummy_text, dummy_lengths),
        str(onnx_path),
        verbose=False,
        do_constant_folding=True,
        opset_version=opset_version,
        input_names=["text", "text_lengths"],
        output_names=["logits"],
    )

    onnx_model = onnx.load(str(onnx_path))
    onnx.checker.check_model(onnx_model)
    print(f"Exported ONNX model to {onnx_path}")
    print(f"Fixed token length for {input_path}: {fixed_length}")
    inspect_onnx(onnx_path)
    return fixed_length


def simplify_onnx(onnx_path: Path, simplified_onnx_path: Path) -> None:
    import onnxsim

    model = onnx.load(str(onnx_path))
    simplified_model, check = onnxsim.simplify(model)
    simplified_onnx_path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(simplified_model, str(simplified_onnx_path))
    print(f"Simplified ONNX saved to {simplified_onnx_path}, check={check}")
    inspect_onnx(simplified_onnx_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--model-revision", type=str, default="v2.0.4")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--onnx-path", type=Path, default=DEFAULT_ONNX_PATH)
    parser.add_argument("--simplified-onnx-path", type=Path, default=DEFAULT_SIMPLIFIED_ONNX_PATH)
    parser.add_argument("--opset-version", type=int, default=14)
    parser.add_argument("--split-size", type=int, default=20)
    parser.add_argument("--step", type=str, default="all", choices=["export", "simplify", "all"])
    args = parser.parse_args()

    if args.step in ["export", "all"]:
        export_onnx(
            model_dir=args.model_dir,
            model_revision=args.model_revision,
            input_path=args.input,
            onnx_path=args.onnx_path,
            opset_version=args.opset_version,
            split_size=args.split_size,
        )

    if args.step in ["simplify", "all"]:
        simplify_onnx(args.onnx_path, args.simplified_onnx_path)


if __name__ == "__main__":
    main()