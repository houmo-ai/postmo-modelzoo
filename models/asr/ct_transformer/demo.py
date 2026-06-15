import argparse
import sys
import os
from pathlib import Path
import numpy as np
import yaml

HOUMO_TARGET = os.getenv("HOUMO_TARGET", "xh2")
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "config.yaml"
DEFAULT_HMM_DIR = Path(__file__).resolve().parent / "output" / HOUMO_TARGET
DEFAULT_MODEL_NAME = "ct_transformer"
DEFAULT_INPUT = Path(__file__).resolve().parent / "ct_transformer/example/punc_example.txt"


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
    repo_ids = model_config.get("modelscope_repo", [])
    if repo_ids:
        return repo_ids[0]
    # Fallback to local directory
    local_dir = Path(__file__).resolve().parent / "ct_transformer"
    if local_dir.exists():
        return str(local_dir)
    return "ct_transformer"

def load_text_items(input_path: Path) -> list[str]:
    items = []
    for line in input_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            items.append(line)
    if not items:
        raise ValueError(f"No valid text lines found in {input_path}")
    return items

def tokenize_to_chunks(auto_model_loader, text: str, split_size: int):
    from funasr.models.ct_transformer.utils import split_to_mini_sentence, split_words
    
    model, tokenizer = auto_model_loader.model, auto_model_loader.kwargs["tokenizer"]
    tokens = split_words(text, jieba_usr_dict=model.jieba_usr_dict)
    token_ids = tokenizer.encode(tokens)
    mini_sentences = split_to_mini_sentence(tokens, split_size)
    mini_sentences_id = split_to_mini_sentence(token_ids, split_size)
    return tokens, mini_sentences, mini_sentences_id

def pad_text_ids(token_ids, fixed_length: int):
    token_ids = np.asarray(token_ids, dtype=np.int32)
    if token_ids.shape[0] > fixed_length:
        raise ValueError(f"Token length {token_ids.shape[0]} exceeds fixed_length {fixed_length}")
    padded = np.zeros((1, fixed_length), dtype=np.int32)
    padded[0, : token_ids.shape[0]] = token_ids
    lengths = np.array([token_ids.shape[0]], dtype=np.int32)
    return padded, lengths

def run_punc_inference_hmm(auto_model_loader, text: str, split_size: int, fixed_length: int, module):
    import copy
    tokens, mini_sentences, mini_sentences_id = tokenize_to_chunks(auto_model_loader, text, split_size)
    punc_model = auto_model_loader.model
    punc_list = punc_model.punc_list
    sentence_end_id = punc_model.sentence_end_id

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

        in_name_0 = module.get_input_name(0)
        module.set_input(in_name_0, padded_text.astype(module.get_input_info(in_name_0).dtype))
        if module.get_num_inputs() > 1:
            in_name_1 = module.get_input_name(1)
            module.set_input(in_name_1, model_text_lengths.astype(module.get_input_info(in_name_1).dtype))

        module.run()
        module.sync()

        out_name = module.get_output_name(0)
        logits_np = module.get_output(out_name).numpy() 
        # logits is numpy array: [1, fixed_length, num_puncs]
        
        # Valid length masking
        valid_logits = logits_np[:, : text_lengths[0], :]
        punctuations = np.argmax(valid_logits, axis=-1).reshape(-1)

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

        new_mini_sentence_punc += [int(item) for item in punctuations]
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

    return new_mini_sentence

def main():
    import time

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--config", type=str, default=str(DEFAULT_CONFIG_PATH), help="path to config.yaml"
    )
    parser.add_argument("--model-dir", type=Path, default=None, help="Paths to original unquantized model for vocabulary/tokenizer processing")
    parser.add_argument("--model-revision", type=str, default="v2.0.4")
    parser.add_argument("--model_name", type=str, default=None, help="model name")
    parser.add_argument("--model_size", type=str, default=None, help="model size")
    parser.add_argument("--input", type=str, default=str(DEFAULT_INPUT))
    parser.add_argument("--split-size", type=int, default=20)
    parser.add_argument("--hmm-dir", type=Path, default=None)
    parser.add_argument("--model-path", type=str, default=None, help="houmo model path")
    args = parser.parse_args()

    default_model_size, default_model_name, model_configs = get_model_configs(
        args.config
    )
    model_name = first_not_none(args.model_name, default_model_name)
    model_size = first_not_none(args.model_size, default_model_size)
    model_config = model_configs.get(model_name, {}).get(model_size, {})

    if args.model_dir is None:
        args.model_dir = Path(get_default_model_dir(model_config))
    if args.hmm_dir is None:
        args.hmm_dir = DEFAULT_HMM_DIR

    t_load0 = time.time()
    from model_utils import load_auto_model
    auto_model_loader = load_auto_model(args.model_dir, args.model_revision)
    
    fixed_length = 500

    try:
        import tcim_lite
        if args.model_path is not None:
            model_path = Path(args.model_path)
        else:
            # Try direct path first, then search in subdirectories (hmatc extraction)
            model_path = args.hmm_dir / f"{model_name}.hmm"
            if not model_path.exists():
                for subdir in args.hmm_dir.iterdir():
                    if subdir.is_dir():
                        candidate = subdir / f"{model_name}.hmm"
                        if candidate.exists():
                            model_path = candidate
                            break
        
        module = tcim_lite.runtime.load(str(model_path))
        print(f"Successfully loaded HMM: {model_path}")
        
        # Override the input dynamically derived fixed_length from memory compiled model shape
        fixed_length = module.get_input_info(module.get_input_name(0)).shape[1]
        print(f"Determined static fixed sequence length from HMM model input shape properties: {fixed_length}")

    except Exception as e:
        print(f"[error] Cannot initialize TCIM module. Ensure you are on NPU environment: {e}")
        return

    t_load1 = time.time()
    text_items = load_text_items(Path(args.input))
    
    results = []
    
    t_inf_total = 0.0
    for idx, text in enumerate(text_items):
        print(f"Processing sequence [{idx}]")
        t_start = time.time()
        final_text = run_punc_inference_hmm(auto_model_loader, text, args.split_size, fixed_length, module)
        t_end = time.time()
        t_inf_total += (t_end - t_start)
        results.append({"text": final_text})
        
    print(results)
    
    from loguru import logger
    logger.success("=" * 100)
    logger.success("                    Model Inference Performance Summary Report")
    logger.success("=" * 100)
    logger.success("Performance Details:")
    logger.success(f"  Load Model+Tokenizer   : {(t_load1 - t_load0) * 1000:>7.2f} ms")
    logger.success(f"  Total Inference ({len(text_items):<2} seqs) : {t_inf_total * 1000:>7.2f} ms")
    if len(text_items) > 0:
        logger.success(f"  Average Inference/seq  : {(t_inf_total * 1000) / len(text_items):>7.2f} ms")
    logger.success("=" * 100)


if __name__ == "__main__":
    main()
