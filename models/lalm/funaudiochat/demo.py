# Copyright (c) 2026 HOUMO AI
#
# File: demo.py
# Description:
#   Command-line entry point for Fun-Audio-Chat Houmo Python Engine.
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

"""Command-line demo for Fun-Audio-Chat text and speech generation."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
IMODELZOO_ROOT = ROOT.parents[2]
sys.path.insert(0, str(IMODELZOO_ROOT / "utils" / "python"))

MODEL_DIR = ROOT / "output" / "xh2"
HMQUANT_DIR = MODEL_DIR / "hmquant"
DEFAULT_STATIC_AUDIO_SAMPLES = 126799
DEFAULT_SYSTEM_PROMPT = "You are asked to generate text tokens."
SPOKEN_PROMPT = (
    "You are asked to generate both text and speech tokens at the same time. "
    "你的名字是小云。你是一位来自杭州的温柔友善的女孩，声音甜美，举止亲切。"
    "你的回复语气自然友好，力求沟通简洁明了。你的回复简短，通常只有一到三句话，"
    "避免使用正式的称谓和重复的短语。你能用恰当的声音回复，遵循用户的指示，"
    "并能共情他们的情绪。你能用恰当的方言回复，会说四川话和粤语。"
)


def _parse_bool(value: str | bool) -> bool:
    """Parse common textual and boolean representations for CLI flags."""
    if isinstance(value, bool):
        return value
    normalized = value.strip().lower()
    if normalized in ("1", "true", "yes", "on"):
        return True
    if normalized in ("0", "false", "no", "off"):
        return False
    raise argparse.ArgumentTypeError(f"expected a boolean value, got {value!r}")


def get_args() -> argparse.ArgumentParser:
    """Build the command-line argument parser for the demo."""
    parser = argparse.ArgumentParser(description="Fun-Audio-Chat S2T, S2S, and VAD-segmented E2E demo")
    parser.add_argument("--stage", dest="stage", type=str, default="s2t", choices=("s2t", "s2s", "e2e"), help="inference pipeline")
    parser.add_argument("--audio_path", dest="audio_path", type=Path, default=None, help="input audio path")
    parser.add_argument("--tokenizer_dir", dest="tokenizer_dir", type=Path, default=ROOT / "Fun-Audio-Chat-8B", help="processor and tokenizer directory")
    parser.add_argument("--embedding_path", dest="embedding_path", type=Path, default=HMQUANT_DIR / "quant_embedding.pt", help="text embedding weights")
    parser.add_argument("--audio_embedding_path", dest="audio_embedding_path", type=Path, default=HMQUANT_DIR / "quant_audio_embedding.pt", help="audio token embedding weights")
    parser.add_argument("--pre_matching_path", dest="pre_matching_path", type=Path, default=HMQUANT_DIR / "audio_decoder_pre_matching.pt", help="audio decoder pre-matching weights")
    parser.add_argument("--flow_input_embedding_path", dest="flow_input_embedding_path", type=Path, default=HMQUANT_DIR / "flow_input_embedding.pt", help="Flow input embedding weights")
    parser.add_argument("--speaker_info_path", dest="speaker_info_path", type=Path, default=HMQUANT_DIR / "new_spk2info.pt", help="default speaker profile")
    parser.add_argument("--audio_encoder_path", dest="audio_encoder_path", type=Path, default=MODEL_DIR / "funaudiochat-8b_audio_encoder.hmm", help="audio encoder HMM")
    parser.add_argument("--prefill_path", dest="prefill_path", type=Path, default=MODEL_DIR / "funaudiochat-8b_prefill.hmm", help="language prefill HMM")
    parser.add_argument("--decode_path", dest="decode_path", type=Path, default=MODEL_DIR / "funaudiochat-8b_decode.hmm", help="language decode HMM")
    parser.add_argument("--audio_tower_path", dest="audio_tower_path", type=Path, default=MODEL_DIR / "funaudiochat-8b_audio_tower.hmm", help="audio tower HMM")
    parser.add_argument("--audio_decoder_prefill_path", dest="audio_decoder_prefill_path", type=Path, default=MODEL_DIR / "funaudiochat-8b_audio_decoder_prefill.hmm", help="CRQ prefill HMM")
    parser.add_argument("--audio_decoder_decode_path", dest="audio_decoder_decode_path", type=Path, default=MODEL_DIR / "funaudiochat-8b_audio_decoder_decode.hmm", help="CRQ decode HMM")
    parser.add_argument("--flow_encoder_path", dest="flow_encoder_path", type=Path, default=MODEL_DIR / "cosyvoice3-0.5b-2512_flow_encoder.hmm", help="Flow encoder HMM")
    parser.add_argument("--flow_spk_path", dest="flow_spk_path", type=Path, default=MODEL_DIR / "cosyvoice3-0.5b-2512_flow_spk.hmm", help="Flow speaker HMM")
    parser.add_argument("--flow_decoder_path", dest="flow_decoder_path", type=Path, default=MODEL_DIR / "cosyvoice3-0.5b-2512_flow_decoder.hmm", help="Flow decoder HMM")
    parser.add_argument("--hift_part1_path", dest="hift_part1_path", type=Path, default=MODEL_DIR / "cosyvoice3-0.5b-2512_hift_part1.hmm", help="HiFT part1 HMM")
    parser.add_argument("--hift_part2_path", dest="hift_part2_path", type=Path, default=MODEL_DIR / "cosyvoice3-0.5b-2512_hift_part2.hmm", help="HiFT part2 HMM")
    parser.add_argument("--vad_path", dest="vad_path", type=Path, default=MODEL_DIR / "fsmd_vad.hmm", help="FSMN VAD HMM")
    parser.add_argument("--config_path", dest="config_path", type=Path, default=HMQUANT_DIR / "config.yaml", help="VAD config")
    parser.add_argument("--cmvn_path", dest="cmvn_path", type=Path, default=HMQUANT_DIR / "am.mvn", help="VAD CMVN")
    parser.add_argument("--system_prompt", dest="system_prompt", type=str, default=None, help="system prompt; default depends on stage")
    parser.add_argument("--max_new_tokens", dest="max_new_tokens", type=int, default=2048, help="maximum generated text tokens")
    parser.add_argument("--temperature", dest="temperature", type=float, default=0.6, help="speech sampling temperature")
    parser.add_argument("--top_k", dest="top_k", type=int, default=20, help="speech sampling top-k")
    parser.add_argument("--top_p", dest="top_p", type=float, default=0.95, help="speech sampling top-p")
    parser.add_argument("--repetition_penalty", dest="repetition_penalty", type=float, default=1.2, help="speech repetition penalty")
    parser.add_argument("--static_audio_samples", dest="static_audio_samples", type=int, default=DEFAULT_STATIC_AUDIO_SAMPLES, help="maximum static 16 kHz input samples")
    parser.add_argument("--device", dest="device", type=int, default=0, help="first NPU device index")
    parser.add_argument("--ndevice", dest="ndevice", type=int, default=1, help="number of NPU devices")
    parser.add_argument("--batch", dest="batch", type=int, default=1, help="batch size; only 1 is supported")
    parser.add_argument("--no_force_audio_bos", dest="no_force_audio_bos", type=_parse_bool, default=False, nargs="?", const=True, help="disable forced text audio BOS")
    parser.add_argument("--token_hop", dest="token_hop", type=int, default=125, help="new codec tokens per token2wav chunk")
    parser.add_argument("--token_overlap", dest="token_overlap", type=int, default=3, help="history codec tokens per non-first chunk")
    parser.add_argument("--fade_ms", dest="fade_ms", type=float, default=5.0, help="chunk edge fade duration")
    parser.add_argument("--seed", dest="seed", type=int, default=42, help="speech and Flow random seed")
    parser.add_argument("--perf", dest="perf", type=_parse_bool, default=True, nargs="?", const=True, help="enable performance reporting; pass false to disable")
    parser.add_argument("--output_wav", dest="output_wav", type=Path, default=ROOT / "result" / "turn_000_response.wav", help="S2S response WAV")
    parser.add_argument("--e2e_output_dir", dest="e2e_output_dir", type=Path, default=ROOT / "result", help="E2E output directory")
    return parser


def _resolve_args(args: argparse.Namespace) -> argparse.Namespace:
    """Resolve defaults that depend on the selected inference stage."""
    if args.audio_path is None:
        examples = Path(os.getenv("HOUMO_EXAMPLES_PATH", str(IMODELZOO_ROOT)))
        args.audio_path = examples / "data" / "audio" / ("questions.wav" if args.stage == "e2e" else "question.wav")
    if args.system_prompt is None:
        args.system_prompt = DEFAULT_SYSTEM_PROMPT if args.stage == "s2t" else SPOKEN_PROMPT
    return args


class HmFunAudioChat:
    """User-facing wrapper that delegates generation to the model engine."""

    def __init__(self, args: argparse.Namespace):
        from funaudiochat_engine import FunAudioChatEngine
        from funaudiochat_types import FunAudioChatPaths

        path_names = [name for name in vars(args) if name.endswith("_path") or name == "tokenizer_dir"]
        paths = FunAudioChatPaths(**{name: getattr(args, name) for name in path_names if name not in ("audio_path", "output_wav")})
        self.engine = FunAudioChatEngine(
            paths,
            stage=args.stage,
            device=args.device,
            ndevice=args.ndevice,
            batch=args.batch,
            static_audio_samples=args.static_audio_samples,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
            force_audio_bos=not args.no_force_audio_bos,
            token_hop=args.token_hop,
            token_overlap=args.token_overlap,
            fade_ms=args.fade_ms,
            seed=args.seed,
            perf=args.perf,
        )

    def generate(self, request, **kwargs):
        yield from self.engine.generate(request, **kwargs)

    def print_perf(self) -> None:
        self.engine.print_perf()


def _validate(args: argparse.Namespace) -> None:
    """Validate argument combinations before loading model resources."""
    if args.max_new_tokens <= 0:
        raise ValueError("--max_new_tokens must be greater than zero")
    if args.ndevice <= 0:
        raise ValueError("--ndevice must be greater than zero")
    if args.batch != 1:
        raise ValueError("--batch must be 1")
    if args.static_audio_samples <= 0:
        raise ValueError("--static_audio_samples must be greater than zero")


def main() -> None:
    """Parse arguments, run the requested pipeline, and print its results."""
    args = _resolve_args(get_args().parse_args())
    _validate(args)
    model = HmFunAudioChat(args)
    from funaudiochat_process import SAMPLE_RATE
    from funaudiochat_types import AudioResult, PerformanceResult, SpeechResult, TextResult, TurnResult, VadResult
    from houmo_engine.perf.formatter import format_report
    from loguru import logger

    result_dir = args.e2e_output_dir.resolve()
    turns = []
    vad_result = None
    streamed_perf = False
    print(f"\033[1;95m\nQ: {args.audio_path}\nA: ", end="", flush=True)
    for event in model.generate(args.audio_path, system_prompt=args.system_prompt, max_new_tokens=args.max_new_tokens):
        if isinstance(event, TextResult):
            print(f"\033[1;95m{event.text}\033[0m")
            print(f"prompt tokens: {event.prompt_tokens}")
            print(f"generated tokens: {event.generated_tokens}")
        elif isinstance(event, SpeechResult):
            print(f"\033[1;95m{event.text}\033[0m")
            print("generate_audio_token_ids:", event.speech_ids)
        elif isinstance(event, AudioResult):
            import torchaudio
            args.output_wav.parent.mkdir(parents=True, exist_ok=True)
            torchaudio.save(str(args.output_wav), event.waveform.cpu(), SAMPLE_RATE)
            print(f"audio saved to: {args.output_wav.resolve()}")
        elif isinstance(event, VadResult):
            print("\033[0m", end="")
            vad_result = event
            result_dir.mkdir(parents=True, exist_ok=True)
            print(json.dumps(event.stats, ensure_ascii=False, indent=2))
            print("vad_segments:", event.segments)
        elif isinstance(event, TurnResult):
            import soundfile as soundfile
            import torchaudio
            input_path = result_dir / f"turn_{event.turn:03d}_input.wav"
            response_path = result_dir / f"turn_{event.turn:03d}_response.wav"
            soundfile.write(input_path, event.input_waveform, vad_result.sample_rate)
            torchaudio.save(str(response_path), event.response_waveform.cpu(), event.sample_rate)
            print("\033[0m", end="")
            print(f"\n=== turn {event.turn}: {event.start_ms} ms - {event.end_ms} ms ===")
            print(f"\033[1;95mgenerate_text: {event.text}\033[0m")
            print(f"audio saved to: {response_path}")
            turns.append({"turn": event.turn, "start_ms": event.start_ms, "end_ms": event.end_ms, "input_wav": str(input_path), "response_text": event.text, "response_wav": str(response_path), "speech_token_count": len(event.speech_ids)})
        elif isinstance(event, PerformanceResult):
            streamed_perf = True
            print("\033[0m", end="")
            sys.stdout.flush()
            logger.opt(raw=True, colors=True).success(
                "<green>\n=== {} performance ===\n{}</green>\n",
                event.label,
                format_report(event.report),
            )
    print("\033[0m", end="")
    if args.stage == "e2e":
        result_path = result_dir / "results.json"
        result_path.write_text(json.dumps({"audio": str(args.audio_path), "vad_stats": vad_result.stats, "segments": vad_result.segments, "turns": turns}, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"e2e results saved to: {result_path}")
    if not streamed_perf:
        model.print_perf()


if __name__ == "__main__":
    main()
