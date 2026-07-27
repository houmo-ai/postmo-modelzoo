import argparse
import copy
import sys
import os
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
import yaml
from torch import nn

from export_onnx import export_onnx, simplify_onnx
from model_utils import freeze_encoder_position_encoder, load_auto_model

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from xhquant.api import DeviceType, QuantScheme, convert_onnx_to_hmonnx, create_quant_config, get_root_logger, xhquant_init
from xhquant.xhonnxruntime.hmonnx_inference import HMONNXInference

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[2]

HOUMO_TARGET = os.getenv("HOUMO_TARGET", "xh2")
DEFAULT_CONFIG_PATH = THIS_DIR / "config.yaml"


def first_not_none(*args):
    """Return the first argument that is not None."""
    for arg in args:
        if arg is not None:
            return arg
    return None


def get_model_configs(config_path: str):
    """Load model configs from yaml file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    default_model_size = config.get("default_model_size", "")
    default_model_name = config.get("default_model_name", "")
    model_configs = config.get("model_configs", {})
    return default_model_size, default_model_name, model_configs


def get_default_model_dir(model_config: dict) -> str:
    """Get the model identifier - use modelscope repo ID so funasr can find cached model."""
    local_dir = Path(__file__).resolve().parent / "ct_transformer"
    if local_dir.exists():
        return str(local_dir)

    repo_ids = model_config.get("modelscope_repo", [])
    if repo_ids:
        return repo_ids[0]
    return "ct_transformer"


DEFAULT_MODEL_DIR = THIS_DIR / "ct_transformer"
DEFAULT_INPUT = DEFAULT_MODEL_DIR / "example" / "punc_example.txt"
DEFAULT_OUTPUT_DIR = THIS_DIR / "work_dirs" / "export_fp32"
DEFAULT_ONNX_PATH = DEFAULT_OUTPUT_DIR / "onnx" / "model.onnx"
DEFAULT_SIMPLIFIED_ONNX_PATH = DEFAULT_OUTPUT_DIR / "onnx" / "model_simplified.onnx"
DEFAULT_OUT_DIR = THIS_DIR / "work_dirs" / "hmonnx"

def load_text_items(input_path: Path) -> list[str]:
    items = []
    for line in input_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            items.append(line)
    if not items:
        raise ValueError(f"No valid text lines found in {input_path}")
    return items


def tokenize_to_chunks(auto_model, text: str, split_size: int):
    from funasr.models.ct_transformer.utils import split_to_mini_sentence, split_words

    tokens = split_words(text, jieba_usr_dict=auto_model.model.jieba_usr_dict)
    token_ids = auto_model.kwargs["tokenizer"].encode(tokens)
    mini_sentences = split_to_mini_sentence(tokens, split_size)
    mini_sentences_id = split_to_mini_sentence(token_ids, split_size)
    return tokens, mini_sentences, mini_sentences_id


def collect_max_chunk_length(auto_model, input_path: Path, split_size: int) -> int:
    max_len = 0
    for text in load_text_items(input_path):
        _, _, mini_sentences_id = tokenize_to_chunks(auto_model, text, split_size)
        cache_sent_id = []
        for mini_sentence_id in mini_sentences_id:
            merged = cache_sent_id + list(mini_sentence_id)
            max_len = max(max_len, len(merged))
            cache_sent_id = []
    if max_len <= 0:
        raise ValueError(f"Failed to derive token length from {input_path}")
    return max_len + 200  # Add buffer for sliding window cache without punctuation


def pad_text_ids(token_ids, fixed_length: int):
    token_ids = np.asarray(token_ids, dtype=np.int32)
    if token_ids.shape[0] > fixed_length:
        raise ValueError(f"Token length {token_ids.shape[0]} exceeds fixed_length {fixed_length}")
    padded = np.zeros((1, fixed_length), dtype=np.int32)
    padded[0, : token_ids.shape[0]] = token_ids
    lengths = np.array([token_ids.shape[0]], dtype=np.int32)
    return padded, lengths


def inspect_onnx(model_path: Path) -> None:
    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    print("Model inputs:")
    for inp in session.get_inputs():
        print(f"  {inp.name}: {inp.type}, shape={inp.shape}")
    print("Model outputs:")
    for out in session.get_outputs():
        print(f"  {out.name}: {out.type}, shape={out.shape}")


def ensure_onnx_artifacts(args, logger):
    export_onnx(
        model_dir=args.model_dir,
        model_revision=args.model_revision,
        input_path=args.input,
        onnx_path=args.onnx_path,
        opset_version=14,
        split_size=args.split_size,
    )
    simplify_onnx(args.onnx_path, args.simplified_onnx_path)
    logger.info(f"Refreshed ONNX artifacts: {args.simplified_onnx_path}")


class TorchPuncWrapper(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model.cpu().float().eval()
        self.chunk_logits = []

    def reset(self):
        self.chunk_logits = []

    def forward(self, text: torch.Tensor, text_lengths: torch.Tensor):
        logits, _ = self.model.punc_forward(text=text, text_lengths=text_lengths)
        self.chunk_logits.append(logits.detach().cpu().float())
        return logits


class OnnxPuncWrapper(nn.Module):
    def __init__(self, onnx_path: Path):
        super().__init__()
        self.session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        self.chunk_logits = []

    def reset(self):
        self.chunk_logits = []

    def forward(self, text: torch.Tensor, text_lengths: torch.Tensor):
        ort_inputs = {"text": text.detach().cpu().numpy().astype(np.int32)}
        if any(item.name == "text_lengths" for item in self.session.get_inputs()):
            ort_inputs["text_lengths"] = text_lengths.detach().cpu().numpy().astype(np.int32)
        outputs = self.session.run(None, ort_inputs)
        logits = torch.from_numpy(outputs[0]).float()
        self.chunk_logits.append(logits.clone())
        return logits


class HmonnxPuncWrapper(nn.Module):
    def __init__(self, hmonnx_path: Path, exec_device: str, save_golden_dir: Path | None = None):
        super().__init__()
        self.session = HMONNXInference(str(hmonnx_path))
        self.session.exec_device = exec_device
        self.session.to(exec_device)
        if save_golden_dir is not None:
            self.session.save_golden = True
            self.session.save_golden_dir = str(save_golden_dir)
        self.input_infos = list(self.session.inputs)
        self.chunk_logits = []

    def reset(self):
        self.chunk_logits = []

    def _to_session_tensor(self, tensor: torch.Tensor, input_index: int) -> torch.Tensor:
        info = self.input_infos[input_index]
        return tensor.to(device=self.session.exec_device, dtype=info.dtype)

    def forward(self, text: torch.Tensor, text_lengths: torch.Tensor):
        input_tensors = [self._to_session_tensor(text.detach().cpu(), 0)]
        if len(self.input_infos) > 1:
            input_tensors.append(self._to_session_tensor(text_lengths.detach().cpu(), 1))
        outputs = self.session.forward(*input_tensors)
        if isinstance(outputs, (tuple, list)):
            logits = outputs[0]
        else:
            logits = outputs
        logits = logits.detach().float().cpu()
        self.chunk_logits.append(logits.clone())
        return logits


def run_punc_inference(auto_model, text: str, split_size: int, fixed_length: int, model_runner):
    tokens, mini_sentences, mini_sentences_id = tokenize_to_chunks(auto_model, text, split_size)
    punc_model = auto_model.model
    punc_list = punc_model.punc_list
    sentence_end_id = punc_model.sentence_end_id

    model_runner.reset()
    cache_sent = []
    cache_sent_id = np.array([], dtype=np.int32)
    new_mini_sentence = ""
    new_mini_sentence_punc = []
    cache_pop_trigger_limit = 200
    punc_array = None

    for mini_sentence_i in range(len(mini_sentences)):
        mini_sentence = list(cache_sent) + list(mini_sentences[mini_sentence_i])
        mini_sentence_id = np.concatenate((cache_sent_id, mini_sentences_id[mini_sentence_i]), axis=0)

        padded_text, text_lengths = pad_text_ids(mini_sentence_id, fixed_length)
        model_text_lengths = np.array([fixed_length], dtype=np.int32)
        text_tensor = torch.from_numpy(padded_text)
        length_tensor = torch.from_numpy(model_text_lengths)
        logits = model_runner.forward(text_tensor, length_tensor)
        valid_logits = logits[:, : text_lengths[0], :]
        punctuations = torch.argmax(valid_logits, dim=-1).reshape(-1)
        assert punctuations.size(0) == len(mini_sentence)

        if mini_sentence_i < len(mini_sentences) - 1:
            sentence_end = -1
            last_comma_index = -1
            for index in range(len(punctuations) - 2, 1, -1):
                if punc_list[int(punctuations[index])] in ["。", "？"]:
                    sentence_end = index
                    break
                if last_comma_index < 0 and punc_list[int(punctuations[index])] == "，":
                    last_comma_index = index

            if sentence_end < 0 and len(mini_sentence) > cache_pop_trigger_limit and last_comma_index >= 0:
                sentence_end = last_comma_index
                punctuations[sentence_end] = sentence_end_id

            cache_sent = mini_sentence[sentence_end + 1 :]
            cache_sent_id = mini_sentence_id[sentence_end + 1 :]
            mini_sentence = mini_sentence[0 : sentence_end + 1]
            punctuations = punctuations[0 : sentence_end + 1]

        punctuations_np = punctuations.cpu().numpy()
        new_mini_sentence_punc += [int(item) for item in punctuations_np]
        words_with_punc = []
        for index in range(len(mini_sentence)):
            if (
                index == 0
                or punc_list[int(punctuations[index - 1])] == "。"
                or punc_list[int(punctuations[index - 1])] == "？"
            ) and len(mini_sentence[index][0].encode()) == 1:
                mini_sentence[index] = mini_sentence[index].capitalize()
            if index == 0 and len(mini_sentence[index][0].encode()) == 1:
                mini_sentence[index] = " " + mini_sentence[index]
            if index > 0 and len(mini_sentence[index][0].encode()) == 1 and len(mini_sentence[index - 1][0].encode()) == 1:
                mini_sentence[index] = " " + mini_sentence[index]
            words_with_punc.append(mini_sentence[index])
            if punc_list[int(punctuations[index])] != "_":
                punc_res = punc_list[int(punctuations[index])]
                if len(mini_sentence[index][0].encode()) == 1:
                    if punc_res == "，":
                        punc_res = ","
                    elif punc_res == "。":
                        punc_res = "."
                    elif punc_res == "？":
                        punc_res = "?"
                words_with_punc.append(punc_res)
        new_mini_sentence += "".join(words_with_punc)

        if mini_sentence_i == len(mini_sentences) - 1 and new_mini_sentence:
            if new_mini_sentence[-1] in ["，", "、"]:
                new_mini_sentence = new_mini_sentence[:-1] + "。"
                new_mini_sentence_punc = new_mini_sentence_punc[:-1] + [sentence_end_id]
            elif new_mini_sentence[-1] == ",":
                new_mini_sentence = new_mini_sentence[:-1] + "."
                new_mini_sentence_punc = new_mini_sentence_punc[:-1] + [sentence_end_id]
            elif new_mini_sentence[-1] not in ["。", "？"] and len(new_mini_sentence[-1].encode()) != 1:
                new_mini_sentence = new_mini_sentence + "。"
                new_mini_sentence_punc = new_mini_sentence_punc[:-1] + [sentence_end_id]
                if len(punctuations):
                    punctuations[-1] = 2
            elif new_mini_sentence[-1] not in [".", "?"] and len(new_mini_sentence[-1].encode()) == 1:
                new_mini_sentence = new_mini_sentence + "."
                new_mini_sentence_punc = new_mini_sentence_punc[:-1] + [sentence_end_id]
                if len(punctuations):
                    punctuations[-1] = 2

        if punc_array is None:
            punc_array = punctuations
        else:
            punc_array = torch.cat([punc_array, punctuations], dim=0)

    if punc_model.jieba_usr_dict is not None and punc_array is not None:
        punc_array = punc_array.reshape(-1)
        len_tokens = len(tokens)
        new_punc_array = copy.copy(punc_array).tolist()
        for index, token in enumerate(tokens[::-1]):
            if "\u0e00" <= token[0] <= "\u9fa5" and len(token) > 1:
                num_append = len(token) - 1
                insert_index = len_tokens - index - 1
                for _ in range(num_append):
                    new_punc_array.insert(insert_index, 1)
        punc_array = torch.tensor(new_punc_array)

    return {"text": new_mini_sentence, "punc_array": punc_array}, [item.numpy() for item in model_runner.chunk_logits]


def max_abs_diff(lhs: np.ndarray, rhs: np.ndarray) -> float:
    return float(np.max(np.abs(lhs - rhs)))


def summarize_diffs(lhs_records, rhs_records) -> tuple[float, float]:
    if len(lhs_records) != len(rhs_records):
        raise ValueError(f"Chunk count mismatch: {len(lhs_records)} vs {len(rhs_records)}")
    chunk_diffs = [max_abs_diff(lhs, rhs) for lhs, rhs in zip(lhs_records, rhs_records, strict=True)]
    flat_lhs = np.concatenate([item.reshape(-1, item.shape[-1]) for item in lhs_records], axis=0)
    flat_rhs = np.concatenate([item.reshape(-1, item.shape[-1]) for item in rhs_records], axis=0)
    return max(chunk_diffs), max_abs_diff(flat_lhs, flat_rhs)


def compare_outputs(lhs, rhs) -> bool:
    lhs_array = lhs["punc_array"].cpu().numpy() if lhs["punc_array"] is not None else None
    rhs_array = rhs["punc_array"].cpu().numpy() if rhs["punc_array"] is not None else None
    return lhs["text"] == rhs["text"] and np.array_equal(lhs_array, rhs_array)


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--config", type=str, default=str(DEFAULT_CONFIG_PATH), help="path to config.yaml"
    )
    parser.add_argument("--model_dir", type=Path, default=None)
    parser.add_argument("--model-revision", type=str, default="v2.0.4")
    parser.add_argument("--model_name", type=str, default=None, help="model name")
    parser.add_argument("--model_size", type=str, default=None, help="model size")
    parser.add_argument("--input", type=Path, default=None)
    parser.add_argument("--onnx-path", type=Path, default=None)
    parser.add_argument("--simplified-onnx-path", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--quant_type", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--split-size", type=int, default=20)
    parser.add_argument("--save-golden", action="store_true")
    parser.add_argument("--no-eval", action="store_true")
    parser.add_argument("--inspect-onnx", action="store_true")
    parser.add_argument("--reuse-onnx", action="store_true")
    args = parser.parse_args()

    default_model_size, default_model_name, model_configs = get_model_configs(
        args.config
    )
    model_name = first_not_none(args.model_name, default_model_name)
    model_size = first_not_none(args.model_size, default_model_size)
    model_config = model_configs.get(model_name, {}).get(model_size, {})

    args.model_dir = first_not_none(
        args.model_dir, Path(get_default_model_dir(model_config))
    )
    args.quant_type = first_not_none(
        args.quant_type, model_config.get("quant_type", "w8a8_sefp")
    )
    args.device = first_not_none(args.device, "cuda:0")
    args.input = first_not_none(args.input, DEFAULT_INPUT)
    args.onnx_path = first_not_none(args.onnx_path, DEFAULT_ONNX_PATH)
    args.simplified_onnx_path = first_not_none(
        args.simplified_onnx_path, DEFAULT_SIMPLIFIED_ONNX_PATH
    )
    args.out_dir = first_not_none(args.out_dir, DEFAULT_OUT_DIR)

    xhquant_init(None, debug=False)
    logger = get_root_logger()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.reuse_onnx:
        if not args.onnx_path.exists():
            raise FileNotFoundError(f"Missing ONNX model: {args.onnx_path}")
        simplify_onnx_if_needed = False
    else:
        simplify_onnx_if_needed = True

    if args.inspect_onnx:
        inspect_onnx(args.onnx_path)

    if simplify_onnx_if_needed:
        ensure_onnx_artifacts(args, logger)
    elif not args.simplified_onnx_path.exists():
        simplify_onnx(args.onnx_path, args.simplified_onnx_path)

    auto_model = load_auto_model(args.model_dir, args.model_revision)
    fixed_length = collect_max_chunk_length(auto_model, args.input, args.split_size)
    freeze_encoder_position_encoder(auto_model.model, fixed_length)
    logger.info(f"Input text file: {args.input}")
    logger.info(f"Fixed token length: {fixed_length}")
    logger.info(f"Model name: {model_name}, Model size: {model_size}")
    logger.info(f"Quant type: {args.quant_type}, Device: {args.device}")

    target_device = DeviceType.XH2a
    quant_scheme = QuantScheme(target_device=target_device, quant_type=args.quant_type)
    quant_config = create_quant_config(quant_scheme)
    
    
    hmonnx_path = args.out_dir / f"hmquant_{model_name}.onnx"
    hmonnx_path.parent.mkdir(parents=True, exist_ok=True)

    if not hmonnx_path.exists():
        logger.info(f"Converting ONNX to hmonnx: {hmonnx_path}")
        dummy_text = torch.zeros((1, fixed_length), dtype=torch.int32)
        dummy_lengths = torch.tensor([fixed_length], dtype=torch.int32)
        session = ort.InferenceSession(str(args.simplified_onnx_path), providers=["CPUExecutionProvider"])
        input_tensors = (dummy_text, dummy_lengths) if len(session.get_inputs()) > 1 else (dummy_text,)
        convert_onnx_to_hmonnx(
            str(args.simplified_onnx_path),
            input_tensors,
            target_device,
            str(hmonnx_path),
            quant_config=quant_config,
            input_names=[item.name for item in session.get_inputs()],
            output_names=["logits"],
        )

    if args.no_eval and not args.save_golden:
        return

    text_items = load_text_items(args.input)
    fp_runner = TorchPuncWrapper(auto_model.model)
    onnx_runner = OnnxPuncWrapper(args.simplified_onnx_path)
    golden_dir = args.out_dir if args.save_golden else None
    
    # Hack HMONNXInference to save dummy numpy names matching the golden naming of ptq scripts: 
    # {prefix}_{input/output}.npz (xhquant generally dumps tensor dicts, but tcim_lite looks for custom format)
    # We will leave XHQuant default dumping format if we specified args.save_golden
    hm_runner = HmonnxPuncWrapper(hmonnx_path, args.device, save_golden_dir=golden_dir)

    fp_outputs = []
    onnx_outputs = []
    hm_outputs = []
    fp_records = []
    onnx_records = []
    hm_records = []

    for text in text_items:
        fp_output, fp_record = run_punc_inference(auto_model, text, args.split_size, fixed_length, fp_runner)
        onnx_output, onnx_record = run_punc_inference(auto_model, text, args.split_size, fixed_length, onnx_runner)
        hm_output, hm_record = run_punc_inference(auto_model, text, args.split_size, fixed_length, hm_runner)
        fp_outputs.append(fp_output)
        onnx_outputs.append(onnx_output)
        hm_outputs.append(hm_output)
        fp_records.extend(fp_record)
        onnx_records.extend(onnx_record)
        hm_records.extend(hm_record)

    if not args.no_eval:
        onnx_chunk_diff, onnx_full_diff = summarize_diffs(fp_records, onnx_records)
        hm_chunk_diff, hm_full_diff = summarize_diffs(fp_records, hm_records)
        for index, text in enumerate(text_items):
            logger.info(f"Source[{index}]: {text}")
            logger.info(f"FP[{index}]: {fp_outputs[index]['text']}")
            logger.info(f"ONNX[{index}]: {onnx_outputs[index]['text']}")
            logger.info(f"HMONNX[{index}]: {hm_outputs[index]['text']}")
            logger.info(
                f"Equal[{index}]: fp-vs-onnx={compare_outputs(fp_outputs[index], onnx_outputs[index])}, "
                f"fp-vs-hmonnx={compare_outputs(fp_outputs[index], hm_outputs[index])}"
            )
        logger.info(f"Logits max abs diff: fp-vs-onnx chunk={onnx_chunk_diff} full={onnx_full_diff}")
        logger.info(f"Logits max abs diff: fp-vs-hmonnx chunk={hm_chunk_diff} full={hm_full_diff}")

    if args.save_golden:
        logger.info(f"Golden saved to {golden_dir}")


if __name__ == "__main__":
    main()