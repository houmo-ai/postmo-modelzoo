import os
import re
import shutil
from pathlib import Path
from hmatc.utils import logger
from xhquant.api import Config
from xhmodel_merak.xh_llm import AutoLLMConfig, AutoLLMModel, format_model_name

_HMQUANT_DIR_RE = re.compile(r"^hmquant_xh2_.+_\d{8}$")


def _build_e_cfg_from_model(args):
    cfg = dict(
        chip_arch="XH2a",
        model=dict(
            model_type="Gemma4ForConditionalGeneration",
            hf_model=args.model,
            model_name=Path(args.model).name,
            context_max_length=args.context_length,
            prefill_chunk_length=args.prefill_chunk_length,
            use_cache=True,
            num_logits_to_keep=1,
            quant_scheme=dict(
                quant_type="w8a8h1_sefp",
            ),
            visual_config=dict(
                export_mode="compact",
                max_size_w=args.max_size_w,
                max_size_h=args.max_size_h,
            ),
            audio_config=dict(
                sampling_rate=args.audio_sampling_rate,
            ),
        ),
    )
    cfg = format_model_name(cfg)
    return Config(cfg)


def move_hmquant_files(output_dir: str | Path, model_name: str) -> None:
    """Move files from hmquant_xh2_{model}_{quant}_{seq}_{context}_{date} to parent directory."""
    output_path = Path(output_dir) / "hmquant"
    model_dir = output_path / model_name
    if not model_dir.exists():
        logger.warning(f"Output directory not found: {model_dir}")
        return

    hmquant_dirs = [
        item
        for item in model_dir.iterdir()
        if item.is_dir() and _HMQUANT_DIR_RE.match(item.name)
    ]
    if not hmquant_dirs:
        logger.info(f"No hmquant directory found in {model_dir}")
        return

    hmquant_dir = hmquant_dirs[0]
    logger.info(f"Moving files from {hmquant_dir.name} to {output_path.name}/")

    for item in hmquant_dir.iterdir():
        target = output_path / item.name

        if item.is_dir():
            if target.exists():
                logger.warning(
                    f"\033[33mOverwriting existing directory: {target}\033[0m"
                )
                shutil.rmtree(target)
            shutil.move(str(item), str(target))
            logger.info(f"  Moved: {item.name}/")
        else:
            if target.exists():
                logger.warning(f"\033[33mOverwriting existing file: {target}\033[0m")
                target.unlink()
            shutil.move(str(item), str(target))
            logger.info(f"  Moved: {item.name}")

    shutil.rmtree(model_dir)
    logger.info(f"Cleanup completed, removed: {hmquant_dir.name}")


def quant_e(args, device, dtype):
    cfg = _build_e_cfg_from_model(args)
    model_name = os.path.basename(args.model).lower()
    work_dir = os.path.join(args.out_dir, "hmquant", model_name)
    os.makedirs(work_dir, exist_ok=True)
    model_cfg = AutoLLMConfig.from_pretrained(cfg.model)
    xh_model = AutoLLMModel.from_pretrained(config=model_cfg)
    xh_model.work_dir = work_dir
    xh_model.export_hmonnx(work_dir)
    move_hmquant_files(args.out_dir, model_name)
