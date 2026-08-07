# Copyright 2025 HOUMO AI
#
# File: lm_runner.py
# Description:
#   XH2 Large Model Execution
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
import copy
import glob
import importlib.util
import json
import os
import shutil
import sys
import yaml
from dataclasses import dataclass
from .utils import logger


# Registry metadata is immutable so workflow selection and override capabilities
# remain stable throughout a quantization run.
@dataclass(frozen=True)
class WorkflowSpec:
    path: str
    override_fields: tuple[str, ...] = ()
    speculative_model_paths: tuple[tuple[str, tuple[str, ...]], ...] = ()


# These fields are workflow parameters. Profile fields select a complete workflow
# and are intentionally excluded from the workflow parameter override list.
COMMON_OVERRIDE_FIELDS = (
    "bits",
    "prefill_chunk_length",
    "context_length",
    "quant_type.llm",
    "quant_type.visual",
)

DEFAULT_METHOD_ORDER = ("gptq", "autoround")
DEFAULT_SPECULATIVE_DECODE = "none"
DEFAULT_ATTENTION = "default"
SPECULATIVE_DECODE_MODES = ("none", "mtp", "dflash")
ATTENTION_MODES = ("default", "flash_attention", "page_attention")

GEMMA4_MTP_MODEL_PATHS = (
    (
        "draft_model_dir",
        ("export", "model", "mtp_config", "assistant_hf_model"),
    ),
    (
        "target_model_dir",
        ("export", "model", "mtp_config", "target_hf_model"),
    ),
)
QWEN_DFLASH_MODEL_PATHS = (
    ("draft_model_dir", ("export", "model", "dflash_config", "hf_model")),
)

WORKFLOW_ROOT = "configs_merak/workflows/xh2a"
LLM_WORKFLOW_ROOT = f"{WORKFLOW_ROOT}/llm_models"


# Profiles are registered only when a complete workflow exists in xh2modelzoo.
# The local key is exactly (quant_method, speculative_decode, attention).
# fmt: off
MODEL_WORKFLOW_REGISTRY = {
    "gemma4": {
        "12b-unified": {
            ("gptq", "none", "default"): WorkflowSpec(f"{LLM_WORKFLOW_ROOT}/gemma4_series/12b_unified/gemma4_12b_unified_full.yaml", COMMON_OVERRIDE_FIELDS),
            ("autoround", "none", "default"): WorkflowSpec(f"{LLM_WORKFLOW_ROOT}/gemma4_series/12b_unified/gemma4_12b_unified_autoround.yaml", COMMON_OVERRIDE_FIELDS),
            ("gptq", "none", "flash_attention"): WorkflowSpec(f"{LLM_WORKFLOW_ROOT}/gemma4_series/12b_unified/gemma4_12b_unified_full_flash_attention.yaml", COMMON_OVERRIDE_FIELDS),
            ("gptq", "mtp", "default"): WorkflowSpec(f"{LLM_WORKFLOW_ROOT}/gemma4_series/12b_unified/gemma4_12b_unified_full_mtp.yaml", COMMON_OVERRIDE_FIELDS, GEMMA4_MTP_MODEL_PATHS),
            ("gptq", "mtp", "page_attention"): WorkflowSpec(f"{LLM_WORKFLOW_ROOT}/gemma4_series/12b_unified/gemma4_12b_unified_full_mtp_page_attention.yaml", COMMON_OVERRIDE_FIELDS, GEMMA4_MTP_MODEL_PATHS),
        },
        "e2b": {
            ("gptq", "none", "default"): WorkflowSpec(f"{LLM_WORKFLOW_ROOT}/gemma4_series/e2b/gemma4_e2b_full.yaml", COMMON_OVERRIDE_FIELDS),
            ("autoround", "none", "default"): WorkflowSpec(f"{LLM_WORKFLOW_ROOT}/gemma4_series/e2b/gemma4_e2b_autoround.yaml", COMMON_OVERRIDE_FIELDS),
            ("gptq", "none", "flash_attention"): WorkflowSpec(f"{LLM_WORKFLOW_ROOT}/gemma4_series/e2b/gemma4_e2b_full_flash_attention.yaml", COMMON_OVERRIDE_FIELDS),
            ("autoround", "none", "flash_attention"): WorkflowSpec(f"{LLM_WORKFLOW_ROOT}/gemma4_series/e2b/gemma4_e2b_autoround_flash_attention.yaml", COMMON_OVERRIDE_FIELDS),
            ("gptq", "mtp", "default"): WorkflowSpec(f"{LLM_WORKFLOW_ROOT}/gemma4_series/e2b/gemma4_e2b_full_mtp.yaml", COMMON_OVERRIDE_FIELDS, GEMMA4_MTP_MODEL_PATHS),
            ("autoround", "mtp", "default"): WorkflowSpec(f"{LLM_WORKFLOW_ROOT}/gemma4_series/e2b/gemma4_e2b_autoround_mtp.yaml", COMMON_OVERRIDE_FIELDS, GEMMA4_MTP_MODEL_PATHS),
            ("gptq", "mtp", "page_attention"): WorkflowSpec(f"{LLM_WORKFLOW_ROOT}/gemma4_series/e2b/gemma4_e2b_full_mtp_page_attention.yaml", COMMON_OVERRIDE_FIELDS, GEMMA4_MTP_MODEL_PATHS),
        },
        "e4b": {
            ("gptq", "none", "default"): WorkflowSpec(f"{LLM_WORKFLOW_ROOT}/gemma4_series/e4b/gemma4_e4b_full.yaml", COMMON_OVERRIDE_FIELDS),
            ("autoround", "none", "default"): WorkflowSpec(f"{LLM_WORKFLOW_ROOT}/gemma4_series/e4b/gemma4_e4b_autoround.yaml", COMMON_OVERRIDE_FIELDS),
            ("gptq", "none", "flash_attention"): WorkflowSpec(f"{LLM_WORKFLOW_ROOT}/gemma4_series/e4b/gemma4_e4b_full_flash_attention.yaml", COMMON_OVERRIDE_FIELDS),
            ("autoround", "none", "flash_attention"): WorkflowSpec(f"{LLM_WORKFLOW_ROOT}/gemma4_series/e4b/gemma4_e4b_autoround_flash_attention.yaml", COMMON_OVERRIDE_FIELDS),
            ("gptq", "mtp", "default"): WorkflowSpec(f"{LLM_WORKFLOW_ROOT}/gemma4_series/e4b/gemma4_e4b_full_mtp.yaml", COMMON_OVERRIDE_FIELDS, GEMMA4_MTP_MODEL_PATHS),
            ("autoround", "mtp", "default"): WorkflowSpec(f"{LLM_WORKFLOW_ROOT}/gemma4_series/e4b/gemma4_e4b_autoround_mtp.yaml", COMMON_OVERRIDE_FIELDS, GEMMA4_MTP_MODEL_PATHS),
            ("gptq", "mtp", "page_attention"): WorkflowSpec(f"{LLM_WORKFLOW_ROOT}/gemma4_series/e4b/gemma4_e4b_full_mtp_page_attention.yaml", COMMON_OVERRIDE_FIELDS, GEMMA4_MTP_MODEL_PATHS),
        },
        "26b-a4b": {
            ("gptq", "none", "default"): WorkflowSpec(f"{LLM_WORKFLOW_ROOT}/gemma4_series/26b_a4b/gemma4_26b_a4b_full.yaml", COMMON_OVERRIDE_FIELDS),
            ("autoround", "none", "default"): WorkflowSpec(f"{LLM_WORKFLOW_ROOT}/gemma4_series/26b_a4b/gemma4_26b_a4b_autoround.yaml", COMMON_OVERRIDE_FIELDS),
            ("gptq", "none", "flash_attention"): WorkflowSpec(f"{LLM_WORKFLOW_ROOT}/gemma4_series/26b_a4b/gemma4_26b_a4b_full_flash_attention.yaml", COMMON_OVERRIDE_FIELDS),
            ("autoround", "none", "flash_attention"): WorkflowSpec(f"{LLM_WORKFLOW_ROOT}/gemma4_series/26b_a4b/gemma4_26b_a4b_autoround_flash_attention.yaml", COMMON_OVERRIDE_FIELDS),
            ("gptq", "mtp", "default"): WorkflowSpec(f"{LLM_WORKFLOW_ROOT}/gemma4_series/26b_a4b/gemma4_26b_a4b_full_mtp.yaml", COMMON_OVERRIDE_FIELDS, GEMMA4_MTP_MODEL_PATHS),
            ("autoround", "mtp", "default"): WorkflowSpec(f"{LLM_WORKFLOW_ROOT}/gemma4_series/26b_a4b/gemma4_26b_a4b_autoround_mtp.yaml", COMMON_OVERRIDE_FIELDS, GEMMA4_MTP_MODEL_PATHS),
            ("gptq", "mtp", "page_attention"): WorkflowSpec(f"{LLM_WORKFLOW_ROOT}/gemma4_series/26b_a4b/gemma4_26b_a4b_full_mtp_page_attention.yaml", COMMON_OVERRIDE_FIELDS, GEMMA4_MTP_MODEL_PATHS),
        },
        "31b": {
            ("gptq", "none", "default"): WorkflowSpec(f"{LLM_WORKFLOW_ROOT}/gemma4_series/31b/gemma4_31b_full.yaml", COMMON_OVERRIDE_FIELDS),
            ("autoround", "none", "default"): WorkflowSpec(f"{LLM_WORKFLOW_ROOT}/gemma4_series/31b/gemma4_31b_autoround.yaml", COMMON_OVERRIDE_FIELDS),
            ("gptq", "none", "flash_attention"): WorkflowSpec(f"{LLM_WORKFLOW_ROOT}/gemma4_series/31b/gemma4_31b_full_flash_attention.yaml", COMMON_OVERRIDE_FIELDS),
            ("autoround", "none", "flash_attention"): WorkflowSpec(f"{LLM_WORKFLOW_ROOT}/gemma4_series/31b/gemma4_31b_autoround_flash_attention.yaml", COMMON_OVERRIDE_FIELDS),
            ("gptq", "mtp", "default"): WorkflowSpec(f"{LLM_WORKFLOW_ROOT}/gemma4_series/31b/gemma4_31b_full_mtp.yaml", COMMON_OVERRIDE_FIELDS, GEMMA4_MTP_MODEL_PATHS),
            ("autoround", "mtp", "default"): WorkflowSpec(f"{LLM_WORKFLOW_ROOT}/gemma4_series/31b/gemma4_31b_autoround_mtp.yaml", COMMON_OVERRIDE_FIELDS, GEMMA4_MTP_MODEL_PATHS),
            ("gptq", "mtp", "page_attention"): WorkflowSpec(f"{LLM_WORKFLOW_ROOT}/gemma4_series/31b/gemma4_31b_full_mtp_page_attention.yaml", COMMON_OVERRIDE_FIELDS, GEMMA4_MTP_MODEL_PATHS),
        },
    },
    "qwen3.5": {
        "0.8b": {
            ("autoround", "none", "default"): WorkflowSpec(f"{LLM_WORKFLOW_ROOT}/qwen3_5/0.8b/qwen3_5_0_8b_full.yaml", COMMON_OVERRIDE_FIELDS),
            ("gptq", "none", "default"): WorkflowSpec(f"{LLM_WORKFLOW_ROOT}/qwen3_5/0.8b/qwen3_5_0_8b_full_gptq.yaml", COMMON_OVERRIDE_FIELDS),
        },
        "2b": {
            ("autoround", "none", "default"): WorkflowSpec(f"{LLM_WORKFLOW_ROOT}/qwen3_5/2b/qwen3_5_2b_full.yaml", COMMON_OVERRIDE_FIELDS),
            ("gptq", "none", "default"): WorkflowSpec(f"{LLM_WORKFLOW_ROOT}/qwen3_5/2b/qwen3_5_2b_full_gptq.yaml", COMMON_OVERRIDE_FIELDS),
        },
        "4b": {
            ("autoround", "none", "default"): WorkflowSpec(f"{LLM_WORKFLOW_ROOT}/qwen3_5/4b/qwen3_5_4b_full.yaml", COMMON_OVERRIDE_FIELDS),
        },
        "9b": {
            ("autoround", "none", "default"): WorkflowSpec(f"{LLM_WORKFLOW_ROOT}/qwen3_5/9b/qwen3_5_9b_full.yaml", COMMON_OVERRIDE_FIELDS),
            ("gptq", "none", "default"): WorkflowSpec(f"{LLM_WORKFLOW_ROOT}/qwen3_5/9b/qwen3_5_9b_full_gptq.yaml", COMMON_OVERRIDE_FIELDS),
            ("autoround", "mtp", "default"): WorkflowSpec(f"{LLM_WORKFLOW_ROOT}/qwen3_5/9b/qwen3_5_9b_full_mtp.yaml", COMMON_OVERRIDE_FIELDS),
            ("gptq", "mtp", "default"): WorkflowSpec(f"{LLM_WORKFLOW_ROOT}/qwen3_5/9b/qwen3_5_9b_full_mtp_gptq.yaml", COMMON_OVERRIDE_FIELDS),
            ("autoround", "dflash", "default"): WorkflowSpec(f"{LLM_WORKFLOW_ROOT}/qwen3_5/9b/qwen3_5_9b_full_dflash.yaml", COMMON_OVERRIDE_FIELDS, QWEN_DFLASH_MODEL_PATHS),
            ("gptq", "dflash", "default"): WorkflowSpec(f"{LLM_WORKFLOW_ROOT}/qwen3_5/9b/qwen3_5_9b_full_dflash_gptq.yaml", COMMON_OVERRIDE_FIELDS, QWEN_DFLASH_MODEL_PATHS),
        },
    },
    "qwen3.6": {
        "27b": {
            ("autoround", "none", "default"): WorkflowSpec(f"{LLM_WORKFLOW_ROOT}/qwen3_5/27b/qwen3_6_27b_full.yaml", COMMON_OVERRIDE_FIELDS),
            ("gptq", "none", "default"): WorkflowSpec(f"{LLM_WORKFLOW_ROOT}/qwen3_5/27b/qwen3_6_27b_full_gptq.yaml", COMMON_OVERRIDE_FIELDS),
            ("autoround", "mtp", "default"): WorkflowSpec(f"{LLM_WORKFLOW_ROOT}/qwen3_5/27b/qwen3_6_27b_full_mtp.yaml", COMMON_OVERRIDE_FIELDS),
            ("gptq", "mtp", "default"): WorkflowSpec(f"{LLM_WORKFLOW_ROOT}/qwen3_5/27b/qwen3_6_27b_full_mtp_gptq.yaml", COMMON_OVERRIDE_FIELDS),
            ("autoround", "dflash", "default"): WorkflowSpec(f"{LLM_WORKFLOW_ROOT}/qwen3_5/27b/qwen3_6_27b_full_dflash.yaml", COMMON_OVERRIDE_FIELDS, QWEN_DFLASH_MODEL_PATHS),
            ("gptq", "dflash", "default"): WorkflowSpec(f"{LLM_WORKFLOW_ROOT}/qwen3_5/27b/qwen3_6_27b_full_dflash_gptq.yaml", COMMON_OVERRIDE_FIELDS, QWEN_DFLASH_MODEL_PATHS),
        },
    },
    "qwen3-tts": {
        "0.6b-customvoice": {
            (None, "none", "default"): WorkflowSpec(f"{WORKFLOW_ROOT}/other_models/qwen3_tts/0_6b_customvoice/qwen3_tts_12hz_0_6b_customvoice.yaml"),
        },
    },
    "mineru2.5-pro-2604": {
        "1.2b": {
            (None, "none", "default"): WorkflowSpec(f"{LLM_WORKFLOW_ROOT}/mineru2_5/mineru2_5_pro_xh2a_4k.yaml"),
        },
    },
}
# fmt: on

BUILD_DEFAULTS = {
    "flash_attention": 2,
    "llm_opt": True,
    "enable_common_subgraph": False,
    "ncore": 2,
    "ndevice": 1,
    "cpp_backend": "v2",
    "all_logits": False,
    "batch": 1,
    "device_kernel_split": 1,
    "prefill_chunk_length": 256,
    "context_length": 2048,
}

BUILD_COMPONENT_FIELDS = set(BUILD_DEFAULTS) | {"type", "enable_build"}
BUILD_COMPONENT_TYPES = {"hmonnx", "prefill", "decode"}

BUILD_BOOLEAN_FIELDS = {
    "llm_opt",
    "enable_common_subgraph",
    "all_logits",
}

BUILD_POSITIVE_INTEGER_FIELDS = {
    "ncore",
    "ndevice",
    "batch",
    "device_kernel_split",
    "prefill_chunk_length",
}


class XH2LmRunner(object):
    def __init__(self, cfg: dict):
        target = cfg.get("target") or os.environ.get("HOUMO_TARGET", "xh2")
        self.target = target
        save_dir = (
            os.environ.get("HMATC_SAVE_DIR")
            or cfg.get("save_dir")
            or "./output"
        )
        if not isinstance(save_dir, str) or not save_dir.strip():
            logger.fatal("save_dir must be a non-empty string")
        self.save_dir = os.path.join(save_dir, target)
        self.build_output_dir = (
            os.environ.get("HMATC_BUILD_OUTPUT_DIR") or self.save_dir
        )
        os.makedirs(self.save_dir, exist_ok=True)
        self.model_cfg = cfg.get("model")
        if not isinstance(self.model_cfg, dict):
            logger.fatal("model must be a mapping")

        self.model_name = self.model_cfg.get("model_name")
        if not isinstance(self.model_name, str) or not self.model_name.strip():
            logger.fatal("model.model_name must be a non-empty string")
        self.model_name = self.model_name.strip()

        self.model_size = self.model_cfg.get("model_size")
        if not isinstance(self.model_size, str) or not self.model_size.strip():
            logger.fatal("model.model_size must be a non-empty string")
        self.model_size = self.model_size.strip()

        self.model_dir = self.model_cfg.get("model_dir")
        if not isinstance(self.model_dir, str) or not self.model_dir.strip():
            logger.fatal("model.model_dir must be a non-empty string")
        self.model_dir = os.path.abspath(os.path.expanduser(self.model_dir))
        if not os.path.isdir(self.model_dir):
            logger.fatal(f"Model directory does not exist: {self.model_dir}")

        self.model_type = self.model_cfg.get("model_type")
        if self.model_type is None:
            self.model_type = self.get_model_type(self.model_dir)
            if self.model_type is None:
                logger.fatal(
                    "Cannot determine model.model_type from model_dir; "
                    "please configure model.model_type as raw or quantized"
                )
        elif self.model_type not in ("raw", "quantized"):
            logger.fatal(
                "model.model_type must be one of: raw, quantized; "
                f"got {self.model_type!r}"
            )

        quant_cfg = cfg.get("quant")
        if quant_cfg is None:
            quant_cfg = {}
        if not isinstance(quant_cfg, dict):
            logger.fatal("quant must be a mapping")
        self.quant_cfg = quant_cfg

        build_cfg = cfg.get("build")
        if build_cfg is None:
            build_cfg = {}
        if not isinstance(build_cfg, dict):
            logger.fatal("build must be a mapping")
        self.build_cfg = build_cfg

        components_cfg = self.build_cfg.get("components")
        if components_cfg is None:
            components_cfg = {}
        if not isinstance(components_cfg, dict):
            logger.fatal("build.components must be a mapping")
        for component_name, component_cfg in components_cfg.items():
            if not isinstance(component_name, str) or not component_name.strip():
                logger.fatal("build.components names must be non-empty strings")
            if not isinstance(component_cfg, dict):
                logger.fatal(f"build.components.{component_name} must be a mapping")
        self.build_components_cfg = components_cfg

    def quant(self, device=None):
        # get() with a default intentionally distinguishes an omitted method
        # (GPTQ) from an explicit YAML null (export only for raw models).
        requested_method = self.quant_cfg.get("method", "gptq")
        if requested_method is not None:
            if not isinstance(requested_method, str):
                logger.fatal("quant.method must be null or a string")
            requested_method = requested_method.strip().lower()
            if requested_method not in DEFAULT_METHOD_ORDER:
                logger.fatal(
                    "quant.method must be one of: null, gptq, autoround; "
                    f"got {requested_method!r}"
                )
        speculative_decode = self.normalize_profile_value(
            self.quant_cfg.get("speculative_decode"),
            "quant.speculative_decode",
            DEFAULT_SPECULATIVE_DECODE,
            SPECULATIVE_DECODE_MODES,
        )
        attention = self.normalize_profile_value(
            self.quant_cfg.get("attention"),
            "quant.attention",
            DEFAULT_ATTENTION,
            ATTENTION_MODES,
        )
        logger.info(
            f"Quantization stage started: model={self.model_name!r}, "
            f"size={self.model_size!r}, model_type={self.model_type!r}, "
            f"method={requested_method!r}, "
            f"speculative_decode={speculative_decode!r}, "
            f"attention={attention!r}, input={self.model_dir!r}, "
            f"output={self.save_dir!r}"
        )

        try:
            selected_method, workflow_spec = self.get_workflow_cfg(
                self.model_name,
                self.model_size,
                requested_method,
                speculative_decode,
                attention,
            )
        except ValueError as exc:
            logger.fatal(str(exc))

        workflow_path = self.get_workflow_path(workflow_spec.path)
        logger.info(
            "Quantization workflow selected: "
            f"requested_method={requested_method!r}, "
            f"selected_method={selected_method!r}, "
            f"speculative_decode={speculative_decode!r}, "
            f"attention={attention!r}, workflow={workflow_path!r}"
        )
        try:
            with open(workflow_path, encoding="utf-8") as file:
                base_workflow = yaml.safe_load(file)
        except (OSError, yaml.YAMLError) as exc:
            logger.fatal(f"Failed to load workflow {workflow_path}: {exc}")
        if not isinstance(base_workflow, dict):
            logger.fatal(f"Workflow must be a mapping: {workflow_path}")

        effective_workflow = copy.deepcopy(base_workflow)
        self.apply_workflow_overrides(
            effective_workflow,
            workflow_spec.override_fields,
        )
        self.apply_speculative_model_overrides(
            effective_workflow,
            workflow_spec.speculative_model_paths,
        )

        # Runtime handling depends on the input artifact, not only the selected
        # base workflow. Existing quantized models use Merak's existing_hf
        # adapter, while raw models either export directly for explicit null or
        # run the named quantizer before export.
        if self.model_type == "quantized":
            runtime_mode = "existing_quantized_model"
            existing_hf_config = {
                "algorithm": "existing_hf",
                "artifact_format": "gptqmodel_hf",
                "existing_hf_model_dir": self.model_dir,
            }
            if selected_method is not None:
                existing_hf_config["method"] = selected_method
            runtime_overrides = {"quant": existing_hf_config}
        elif requested_method is None:
            runtime_mode = "export_only"
            runtime_overrides = {"quant": None}
        else:
            runtime_mode = "quantize_and_export"
            if not isinstance(effective_workflow.get("quant"), dict):
                logger.fatal(
                    "The selected workflow does not configure quantization; "
                    "set quant.method to null to export the raw model"
                )
            runtime_overrides = None

        # Persist the final quant state consumed by AutoWorkflow so the effective
        # file describes this run rather than only the selected base workflow.
        if runtime_overrides is not None:
            effective_workflow["quant"] = copy.deepcopy(runtime_overrides["quant"])

        effective_workflow_path = os.path.join(self.save_dir, "effective_workflow.yaml")
        try:
            with open(effective_workflow_path, "w", encoding="utf-8") as file:
                yaml.safe_dump(effective_workflow, file, sort_keys=False)
        except OSError as exc:
            logger.fatal(
                f"Failed to save effective workflow {effective_workflow_path}: {exc}"
            )
        logger.info(
            f"Effective quantization workflow saved: {effective_workflow_path!r}"
        )

        device = device or self.get_device()
        logger.info(
            f"Quantization runtime selected: mode={runtime_mode!r}, "
            f"device={device!r}"
        )

        gptqmodel_spec = importlib.util.find_spec("gptqmodel")
        if gptqmodel_spec is None or gptqmodel_spec.origin is None:
            logger.fatal("Cannot determine GPTQModel root from the gptqmodel package")
        gptqmodel_root = os.path.dirname(
            os.path.dirname(os.path.abspath(gptqmodel_spec.origin))
        )
        autoround_path = os.path.join(gptqmodel_root, "third_party", "auto-round")
        if not os.path.isdir(autoround_path):
            logger.fatal(f"AutoRound directory does not exist: {autoround_path}")
        pythonpath = os.environ.get("PYTHONPATH")
        os.environ["PYTHONPATH"] = os.pathsep.join(
            [autoround_path] + ([pythonpath] if pythonpath else [])
        )
        if autoround_path not in sys.path:
            sys.path.insert(0, autoround_path)

        try:
            from xhmodel_merak.workflows import AutoWorkflow

            # from xhmodel_merak.xh_llm.workflows.auto import AutoLLMWorkflow as AutoWorkflow
        except ImportError as exc:
            logger.fatal(f"Failed to import Merak large-model workflow: {exc}")

        workflow = AutoWorkflow.from_config(
            model_dir=self.model_dir,
            config_path=effective_workflow_path,
        )
        quant_result = workflow.quant(
            output_dir=os.path.join(self.save_dir, "quantized_model"),
            device=device,
            config_overrides=runtime_overrides,
        )
        export_dir = os.path.join(self.save_dir, "hmquant", "export")
        shutil.rmtree(os.path.dirname(export_dir), ignore_errors=True)
        export_result = workflow.export(
            quant_result=quant_result,
            output_dir=export_dir,
            device=device,
            config_overrides=runtime_overrides,
        )
        export_result = self.move_export_artifacts(export_result, export_dir)
        logger.info(
            f"Quantization stage completed: artifacts={export_result.work_dir!r}, "
            f"config={export_result.config_file!r}"
        )
        return export_result

    def move_export_artifacts(self, export_result, export_dir):
        export_dir = os.path.abspath(export_dir)
        work_dir = os.path.abspath(export_result.work_dir)
        if work_dir == export_dir:
            # Some Merak exporters report the requested output directory even
            # though the actual artifacts are placed in one hmquant_* child.
            artifact_dirs = [
                entry.path
                for entry in os.scandir(export_dir)
                if entry.is_dir() and entry.name.startswith("hmquant_")
            ]
            if len(artifact_dirs) != 1:
                logger.fatal(
                    "Expected exactly one HMQuant artifact directory under "
                    f"{export_dir}, found: {artifact_dirs}"
                )
            work_dir = artifact_dirs[0]

        try:
            work_dir_in_export = (
                os.path.commonpath((export_dir, work_dir)) == export_dir
            )
        except ValueError:
            work_dir_in_export = False
        if not work_dir_in_export:
            logger.fatal(
                f"Export work directory is not under the expected export directory: {work_dir}"
            )
        if not os.path.isdir(work_dir):
            logger.fatal(f"Export work directory does not exist: {work_dir}")

        hmquant_dir = os.path.dirname(export_dir)
        entries = os.listdir(work_dir)
        conflicts = [
            name for name in entries if os.path.exists(os.path.join(hmquant_dir, name))
        ]
        if conflicts:
            logger.warning(
                "Overwriting existing HMQuant export artifacts: " f"{conflicts}"
            )
            for name in conflicts:
                destination = os.path.join(hmquant_dir, name)
                if os.path.isdir(destination) and not os.path.islink(destination):
                    shutil.rmtree(destination, ignore_errors=True)
                else:
                    os.remove(destination)

        config_file = os.path.abspath(export_result.config_file)
        try:
            config_in_work_dir = os.path.commonpath((work_dir, config_file)) == work_dir
        except ValueError:
            config_in_work_dir = False

        for name in entries:
            shutil.move(
                os.path.join(work_dir, name),
                os.path.join(hmquant_dir, name),
            )
        shutil.rmtree(export_dir, ignore_errors=True)

        export_result.work_dir = hmquant_dir
        if config_in_work_dir:
            export_result.config_file = os.path.join(
                hmquant_dir,
                os.path.relpath(config_file, work_dir),
            )
        return export_result

    @staticmethod
    def get_workflow_path(workflow_cfg):
        try:
            import xhmodel_merak
        except ImportError as exc:
            logger.fatal(f"Failed to import xhmodel_merak: {exc}")

        package_file = getattr(xhmodel_merak, "__file__", None)
        if not package_file:
            logger.fatal("Cannot determine xh2modelzoo root from xhmodel_merak")
        modelzoo_root = os.path.dirname(os.path.dirname(os.path.abspath(package_file)))

        # Registry entries are complete paths relative to this root, so path
        # resolution is intentionally a direct join without prefix rewriting.
        workflow_path = os.path.join(modelzoo_root, workflow_cfg)
        if not os.path.isfile(workflow_path):
            logger.fatal(f"Workflow configuration does not exist: {workflow_path}")
        return workflow_path

    def apply_workflow_overrides(self, workflow, override_fields):
        # Profile fields are accepted by the user schema but consumed during
        # registry selection or by the dedicated speculative-model helper.
        supported_fields = {
            "method",
            "speculative_decode",
            "attention",
            "speculative_model",
            "bits",
            "prefill_chunk_length",
            "context_length",
            "quant_type",
        }
        unsupported_fields = sorted(set(self.quant_cfg) - supported_fields)
        if unsupported_fields:
            logger.fatal(f"Unsupported quant fields: {unsupported_fields}")

        allowed_fields = set(override_fields)
        overrides = (
            ("bits", ("quant", "bits")),
            (
                "prefill_chunk_length",
                ("export", "model", "prefill_chunk_length"),
            ),
            ("context_length", ("export", "model", "context_max_length")),
        )
        for config_key, workflow_path in overrides:
            if config_key not in self.quant_cfg:
                continue
            if config_key not in allowed_fields:
                logger.fatal(
                    f"quant.{config_key} cannot be overridden for the selected workflow"
                )
            self.set_existing_workflow_value(
                workflow,
                workflow_path,
                self.quant_cfg[config_key],
                f"quant.{config_key}",
            )

        quant_type = self.quant_cfg.get("quant_type")
        if quant_type is None:
            return
        if not isinstance(quant_type, dict):
            logger.fatal("quant.quant_type must be a mapping")

        supported_quant_types = {"llm", "visual"}
        unsupported_quant_types = sorted(set(quant_type) - supported_quant_types)
        if unsupported_quant_types:
            logger.fatal(
                "Unsupported quant.quant_type fields: " f"{unsupported_quant_types}"
            )

        if "llm" in quant_type:
            if "quant_type.llm" not in allowed_fields:
                logger.fatal(
                    "quant.quant_type.llm cannot be overridden for the selected workflow"
                )
            self.set_existing_workflow_value(
                workflow,
                ("export", "model", "quant_scheme", "quant_type"),
                quant_type["llm"],
                "quant.quant_type.llm",
            )
        if "visual" in quant_type:
            if "quant_type.visual" not in allowed_fields:
                logger.fatal(
                    "quant.quant_type.visual cannot be overridden for the selected workflow"
                )
            visual_paths = (
                ("export", "model", "visual_config", "quant_scheme", "quant_type"),
                (
                    "export",
                    "model",
                    "video_visual_config",
                    "quant_scheme",
                    "quant_type",
                ),
            )
            for workflow_path in visual_paths:
                self.get_existing_workflow_parent(
                    workflow, workflow_path, "quant.quant_type.visual"
                )
            for workflow_path in visual_paths:
                self.set_existing_workflow_value(
                    workflow,
                    workflow_path,
                    quant_type["visual"],
                    "quant.quant_type.visual",
                )

    @staticmethod
    def get_existing_workflow_parent(workflow, path, config_name):
        current = workflow
        for part in path[:-1]:
            if not isinstance(current, dict) or part not in current:
                logger.fatal(
                    f"Cannot override {config_name}; workflow path does not exist: "
                    f"{'.'.join(path)}"
                )
            current = current[part]
        if not isinstance(current, dict) or path[-1] not in current:
            logger.fatal(
                f"Cannot override {config_name}; workflow path does not exist: "
                f"{'.'.join(path)}"
            )
        return current

    @classmethod
    def set_existing_workflow_value(cls, workflow, path, value, config_name):
        parent = cls.get_existing_workflow_parent(workflow, path, config_name)
        parent[path[-1]] = value

    @staticmethod
    def normalize_profile_value(value, config_name, default, supported):
        if value is None:
            return default
        if not isinstance(value, str):
            logger.fatal(f"{config_name} must be null or a string")
        normalized = value.strip().lower()
        if normalized not in supported:
            supported_values = ", ".join(supported)
            logger.fatal(f"{config_name} must be one of: {supported_values}")
        return normalized

    @staticmethod
    def normalize_model_directory(value, config_name):
        if not isinstance(value, str) or not value.strip():
            logger.fatal(f"{config_name} must be a non-empty string")
        path = os.path.abspath(os.path.expanduser(value.strip()))
        if not os.path.isdir(path):
            logger.fatal(f"{config_name} must be an existing directory: {path}")
        return path

    def apply_speculative_model_overrides(self, workflow, path_mappings):
        configured_speculative_model = self.quant_cfg.get("speculative_model")
        if configured_speculative_model is not None and not isinstance(
            configured_speculative_model, dict
        ):
            logger.fatal("quant.speculative_model must be a mapping")

        if not path_mappings:
            if configured_speculative_model is not None:
                logger.fatal(
                    "Selected workflow profile does not support "
                    "quant.speculative_model"
                )
            return

        speculative_model = configured_speculative_model or {}
        mapping_by_field = dict(path_mappings)
        unsupported_fields = sorted(set(speculative_model) - set(mapping_by_field))
        if unsupported_fields:
            logger.fatal(
                "Unsupported quant.speculative_model fields: " f"{unsupported_fields}"
            )

        for field_name, workflow_path in path_mappings:
            config_name = f"quant.speculative_model.{field_name}"
            parent = self.get_existing_workflow_parent(
                workflow,
                workflow_path,
                config_name,
            )
            current_value = parent[workflow_path[-1]]
            if field_name in speculative_model:
                parent[workflow_path[-1]] = self.normalize_model_directory(
                    speculative_model[field_name],
                    config_name,
                )
            elif current_value is None or (
                isinstance(current_value, str) and not current_value.strip()
            ):
                logger.fatal(f"{config_name} is required")

    @staticmethod
    def get_device():
        try:
            import torch
        except ImportError as exc:
            logger.fatal(f"Failed to import torch: {exc}")
        if torch.cuda.is_available():
            return "cuda"
        logger.warning(
            "CUDA is not available; quantization and export will run on CPU and may be very slow"
        )
        return "cpu"

    @staticmethod
    def get_model_type(model_dir):
        """Return a recognized model type, or None when it is ambiguous."""
        config_path = os.path.join(model_dir, "config.json")
        if not os.path.isfile(config_path):
            return None

        try:
            with open(config_path, encoding="utf-8") as file:
                config = json.load(file)
        except (OSError, json.JSONDecodeError):
            return None

        if not isinstance(config, dict):
            return None

        quantization_config = config.get("quantization_config")
        if isinstance(quantization_config, dict) and quantization_config:
            return "quantized"
        return "raw"

    @staticmethod
    def get_workflow_cfg(
        model_name,
        model_size,
        method,
        speculative_decode=DEFAULT_SPECULATIVE_DECODE,
        attention=DEFAULT_ATTENTION,
    ):
        try:
            profiles = MODEL_WORKFLOW_REGISTRY[model_name][model_size]
        except KeyError as exc:
            raise ValueError(
                XH2LmRunner.get_unsupported_workflow_message(
                    model_name,
                    model_size,
                    method,
                    speculative_decode,
                    attention,
                )
            ) from exc

        if method is not None:
            profile = (method, speculative_decode, attention)
            if profile in profiles:
                return method, profiles[profile]
        else:
            default_profile = (None, speculative_decode, attention)
            if default_profile in profiles:
                return None, profiles[default_profile]

            for fallback_method in DEFAULT_METHOD_ORDER:
                fallback_profile = (
                    fallback_method,
                    speculative_decode,
                    attention,
                )
                if fallback_profile in profiles:
                    return fallback_method, profiles[fallback_profile]

        raise ValueError(
            XH2LmRunner.get_unsupported_workflow_message(
                model_name,
                model_size,
                method,
                speculative_decode,
                attention,
            )
        )

    @staticmethod
    def get_unsupported_workflow_message(
        model_name,
        model_size,
        method,
        speculative_decode=DEFAULT_SPECULATIVE_DECODE,
        attention=DEFAULT_ATTENTION,
    ):
        model_registry = MODEL_WORKFLOW_REGISTRY.get(model_name)
        if model_registry is None:
            return (
                f"Unsupported model_name={model_name!r}. "
                f"Supported model names: {sorted(MODEL_WORKFLOW_REGISTRY)}"
            )

        profiles = model_registry.get(model_size)
        if profiles is None:
            return (
                f"Unsupported model_size={model_size!r} for "
                f"model_name={model_name!r}. "
                f"Supported model sizes: {sorted(model_registry)}"
            )

        supported_profiles = sorted(
            profiles,
            key=lambda profile: (
                "" if profile[0] is None else profile[0],
                profile[1],
                profile[2],
            ),
        )
        supported_lines = "\n".join(
            "  method="
            f"{profile_method!r}, speculative_decode={profile_speculative!r}, "
            f"attention={profile_attention!r}"
            for profile_method, profile_speculative, profile_attention in (
                supported_profiles
            )
        )
        return (
            f"Unsupported workflow profile for model {model_name}/{model_size}:\n"
            f"  method={method!r}\n"
            f"  speculative_decode={speculative_decode!r}\n"
            f"  attention={attention!r}\n\n"
            f"Supported profiles:\n{supported_lines}"
        )

    def discover_build_components(self):
        hmquant_dir = os.path.join(self.save_dir, "hmquant")
        if not os.path.isdir(hmquant_dir):
            logger.fatal(
                f"HMQuant directory does not exist: {hmquant_dir}; "
                "run quant/export first or check save_dir and target"
            )

        components = {}
        # Direct-child discovery keeps component names open-ended and avoids
        # depending on model-specific golden_meta_info.json schemas.
        for component_name in sorted(os.listdir(hmquant_dir)):
            component_dir = os.path.join(hmquant_dir, component_name)
            if not os.path.isdir(component_dir):
                continue
            hmonnx = self.find_component_hmonnx(component_dir)
            if hmonnx is not None:
                components[component_name] = hmonnx

        if not components:
            logger.fatal(f"No HMONNX components found under {hmquant_dir}")

        unknown_components = sorted(set(self.build_components_cfg) - set(components))
        if unknown_components:
            logger.fatal(
                f"Unknown build components: {unknown_components}. "
                f"Available components: {sorted(components)}"
            )
        return components

    @staticmethod
    def find_component_hmonnx(component_dir):
        patterns = ("*_with_act.onnx", "hmquant_*.onnx", "*.onnx")
        for pattern in patterns:
            matches = sorted(
                path
                for path in glob.glob(os.path.join(component_dir, pattern))
                if os.path.isfile(path)
            )
            if len(matches) > 1:
                logger.fatal(
                    f"Multiple HMONNX files match {pattern!r} under "
                    f"{component_dir}: {matches}"
                )
            if matches:
                return os.path.abspath(matches[0])
        return None

    def resolve_component_build_config(self, component_name, inferred_type):
        top_level_fields = set(self.build_cfg) - {"components"}
        unknown_top_level = sorted(top_level_fields - set(BUILD_DEFAULTS))
        if unknown_top_level:
            logger.fatal(
                f"Unsupported build fields: {unknown_top_level}. "
                f"Supported fields: {sorted(BUILD_DEFAULTS)}"
            )

        component_cfg = self.build_components_cfg.get(component_name, {})
        unknown_component_fields = sorted(set(component_cfg) - BUILD_COMPONENT_FIELDS)
        if unknown_component_fields:
            logger.fatal(
                f"Unsupported fields in build.components.{component_name}: "
                f"{unknown_component_fields}. Supported fields: "
                f"{sorted(BUILD_COMPONENT_FIELDS)}"
            )

        resolved = copy.deepcopy(BUILD_DEFAULTS)
        resolved["enable_build"] = component_cfg.get("enable_build", True)
        for field in BUILD_DEFAULTS:
            if field in self.build_cfg:
                resolved[field] = self.build_cfg[field]
            if field in component_cfg:
                resolved[field] = component_cfg[field]
        explicit_type = component_cfg.get("type")
        if explicit_type is None:
            if inferred_type is None:
                logger.fatal(
                    f"Cannot infer build type for component {component_name!r}: "
                    "the HMONNX contains KV Cache nodes, but its LLM feature "
                    "sequence length cannot be determined. Configure "
                    f"build.components.{component_name}.type as prefill or decode"
                )
            resolved["type"] = inferred_type
        else:
            resolved["type"] = explicit_type
            if inferred_type is not None:
                normalized_type = (
                    explicit_type.strip().lower()
                    if isinstance(explicit_type, str)
                    else explicit_type
                )
                if normalized_type != inferred_type:
                    logger.fatal(
                        f"build component {component_name!r} type is configured "
                        f"as {explicit_type!r}, but its HMONNX graph was detected "
                        f"as {inferred_type!r}"
                    )
            else:
                normalized_type = (
                    explicit_type.strip().lower()
                    if isinstance(explicit_type, str)
                    else explicit_type
                )
                if normalized_type == "hmonnx":
                    logger.fatal(
                        f"Cannot infer whether build component {component_name!r} "
                        "is prefill or decode from its HMONNX graph; configure "
                        f"build.components.{component_name}.type as prefill or decode"
                    )
                logger.warning(
                    f"Cannot verify the configured type for component "
                    f"{component_name!r} from its HMONNX graph; using "
                    f"type={explicit_type!r}"
                )
        self.validate_component_build_config(component_name, resolved)
        return resolved

    @staticmethod
    def validate_component_build_config(component_name, config):
        if not isinstance(config["enable_build"], bool):
            logger.fatal(
                f"build component {component_name!r} field enable_build "
                "must be a boolean"
            )
        for field in BUILD_BOOLEAN_FIELDS:
            if not isinstance(config[field], bool):
                logger.fatal(
                    f"build component {component_name!r} field {field} "
                    "must be a boolean"
                )
        component_type = config["type"]
        if not isinstance(component_type, str):
            logger.fatal(f"build component {component_name!r} type must be a string")
        component_type = component_type.strip().lower()
        if component_type not in BUILD_COMPONENT_TYPES:
            logger.fatal(
                f"build component {component_name!r} type must be one of: "
                f"{sorted(BUILD_COMPONENT_TYPES)}"
            )
        config["type"] = component_type

        flash_attention = config["flash_attention"]
        if type(flash_attention) is not int or flash_attention not in (0, 1, 2):
            logger.fatal(
                f"build component {component_name!r} flash_attention "
                "must be one of: 0, 1, 2"
            )
        for field in BUILD_POSITIVE_INTEGER_FIELDS:
            value = config[field]
            if type(value) is not int or value <= 0:
                logger.fatal(
                    f"build component {component_name!r} field {field} "
                    "must be a positive integer"
                )
        context_length = config["context_length"]
        if config["type"] in {"prefill", "decode"} and context_length is not None:
            if type(context_length) is not int or context_length <= 0:
                logger.fatal(
                    f"build component {component_name!r} field context_length "
                    "must be null or a positive integer"
                )
        if config["ndevice"] not in (1, 2, 4):
            logger.fatal(
                f"build component {component_name!r} ndevice must be " "one of: 1, 2, 4"
            )
        if (
            not isinstance(config["cpp_backend"], str)
            or not config["cpp_backend"].strip()
        ):
            logger.fatal(
                f"build component {component_name!r} cpp_backend must be "
                "a non-empty string"
            )
        config["cpp_backend"] = config["cpp_backend"].strip()

        if config["type"] == "decode" and config["batch"] != 1:
            logger.fatal(
                f"build component {component_name!r} is decode, so batch "
                "must equal 1"
            )

    @staticmethod
    def inspect_hmonnx(hmonnx):
        try:
            import onnx
        except ImportError as exc:
            logger.fatal(f"Failed to import onnx for inspecting HMONNX {hmonnx}: {exc}")

        try:
            model = onnx.load(hmonnx, load_external_data=False)
        except Exception as exc:
            logger.fatal(f"Failed to inspect HMONNX {hmonnx}: {exc}")

        has_kv_cache = False
        has_sliding_window_attention = False
        for node in model.graph.node:
            if node.op_type not in ("KVcache", "KVCacheProcess"):
                continue
            has_kv_cache = True
            for attr in node.attribute:
                if attr.name != "attention_max_length":
                    continue
                try:
                    attention_max_length = int(onnx.helper.get_attribute_value(attr))
                except (TypeError, ValueError) as exc:
                    logger.fatal(
                        f"Invalid attention_max_length in HMONNX {hmonnx} "
                        f"node {node.name!r}: {exc}"
                    )
                if attention_max_length > 0:
                    has_sliding_window_attention = True

        # A graph without KV Cache is an ordinary HMONNX component. LLM graphs
        # are classified from the unique rank-3 feature input: sequence length
        # one is decode, while a larger static length is prefill. Ambiguous
        # graphs remain unclassified so callers must provide an explicit type.
        if not has_kv_cache:
            return {
                "type": "hmonnx",
                "has_sliding_window_attention": False,
            }

        feature_inputs = []
        for graph_input in model.graph.input:
            tensor_type = graph_input.type.tensor_type
            if not tensor_type.HasField("shape") or len(tensor_type.shape.dim) != 3:
                continue
            feature_inputs.append(graph_input)

        inferred_type = None
        if len(feature_inputs) == 1:
            sequence_dim = feature_inputs[0].type.tensor_type.shape.dim[1]
            if sequence_dim.HasField("dim_value") and sequence_dim.dim_value > 0:
                inferred_type = "decode" if sequence_dim.dim_value == 1 else "prefill"

        return {
            "type": inferred_type,
            "has_sliding_window_attention": has_sliding_window_attention,
        }

    @staticmethod
    def get_component_build_kwargs(config, prefill_length=None):
        build_kwargs = {
            "ncore": config["ncore"],
            "ndevice": config["ndevice"],
            "cpp_backend": config["cpp_backend"],
            "enable_common_subgraph": config["enable_common_subgraph"],
            "device_kernel_split": config["device_kernel_split"],
        }
        if config["type"] == "prefill":
            build_kwargs.update(
                {
                    "batch": config["batch"],
                    "flash_attn": config["flash_attention"],
                    "llm_opt": config["llm_opt"],
                    "all_logits": config["all_logits"],
                    "prefill_length": prefill_length,
                    "context_length": config["context_length"],
                    "is_prefill": True,
                }
            )
        elif config["type"] == "decode":
            build_kwargs.update(
                {
                    "batch": 1,
                    "llm_batch": config["batch"],
                    "flash_attn": config["flash_attention"],
                    "llm_opt": config["llm_opt"],
                    "all_logits": config["all_logits"],
                    "context_length": config["context_length"],
                    "is_prefill": False,
                }
            )
        else:
            build_kwargs.update(
                {
                    "batch": config["batch"],
                    "flash_attn": config["flash_attention"],
                }
            )
        return build_kwargs

    @staticmethod
    def get_effective_component_config(
        config,
        hmonnx,
        hmm_path,
        prefill_length=None,
    ):
        effective = {
            "enable_build": config["enable_build"],
            "hmonnx": hmonnx,
            "hmm": hmm_path,
            "type": config["type"],
            "flash_attention": config["flash_attention"],
            "enable_common_subgraph": config["enable_common_subgraph"],
            "ncore": config["ncore"],
            "ndevice": config["ndevice"],
            "cpp_backend": config["cpp_backend"],
            "batch": config["batch"],
            "device_kernel_split": config["device_kernel_split"],
        }
        if config["type"] in ("prefill", "decode"):
            effective.update(
                {
                    "llm_opt": config["llm_opt"],
                    "all_logits": config["all_logits"],
                    "context_length": config["context_length"],
                }
            )
        if config["type"] == "prefill":
            effective["prefill_chunk_length"] = prefill_length
        return effective

    def build(self):
        try:
            from .exec.xh2_exec import Xh2Exec
        except ImportError as exc:
            logger.fatal(
                f"Failed to import XH2 build support; please install tcim: {exc}"
            )

        os.makedirs(self.build_output_dir, exist_ok=True)
        hmquant_dir = os.path.join(self.save_dir, "hmquant")
        logger.info(
            f"Build stage started: model={self.model_name!r}, "
            f"size={self.model_size!r}, target={self.target!r}, "
            f"input={hmquant_dir!r}, output={self.build_output_dir!r}"
        )
        components = self.discover_build_components()
        logger.info(
            f"Build components discovered: count={len(components)}, "
            f"components={components!r}"
        )
        resolved_components = {}
        build_calls = {}
        skipped_components = []
        for component_name, hmonnx in components.items():
            component_cfg = self.build_components_cfg.get(component_name, {})
            enable_build = component_cfg.get("enable_build", True)
            if not isinstance(enable_build, bool):
                logger.fatal(
                    f"build component {component_name!r} field enable_build "
                    "must be a boolean"
                )
            if not enable_build:
                unknown_component_fields = sorted(
                    set(component_cfg) - BUILD_COMPONENT_FIELDS
                )
                if unknown_component_fields:
                    logger.fatal(
                        f"Unsupported fields in build.components.{component_name}: "
                        f"{unknown_component_fields}. Supported fields: "
                        f"{sorted(BUILD_COMPONENT_FIELDS)}"
                    )
                resolved_components[component_name] = {
                    "enable_build": False,
                    "hmonnx": hmonnx,
                    "hmm": None,
                }
                skipped_components.append(component_name)
                logger.info(
                    f"Build component skipped: name={component_name!r}, "
                    "reason='enable_build is false'"
                )
                continue

            inspection = self.inspect_hmonnx(hmonnx)
            config = self.resolve_component_build_config(
                component_name,
                inspection["type"],
            )
            prefill_length = None
            if config["type"] == "prefill":
                # Keeping this value as None prevents the compiler from applying
                # fill-length modification, which is unsupported together with
                # sliding-window attention.
                if inspection["has_sliding_window_attention"]:
                    logger.warning(
                        f"Sliding-window attention was detected for component "
                        f"{component_name!r}; configured prefill_chunk_length="
                        f"{config['prefill_chunk_length']} will not be passed to "
                        "the compiler"
                    )
                else:
                    prefill_length = config["prefill_chunk_length"]
            hmm_name = f"{self.model_name}-{self.model_size}_{component_name}"
            hmm_path = os.path.join(
                self.build_output_dir,
                f"{hmm_name}.hmm",
            )
            resolved_components[component_name] = self.get_effective_component_config(
                config,
                hmonnx,
                hmm_path,
                prefill_length,
            )
            build_calls[component_name] = {
                "hmonnx": hmonnx,
                "hmm_name": hmm_name,
                "output": self.build_output_dir,
                "target": self.target,
                **self.get_component_build_kwargs(config, prefill_length),
            }
            logger.info(
                f"Build component resolved: name={component_name!r}, "
                f"type={config['type']!r}, source={hmonnx!r}, "
                f"artifact={hmm_path!r}"
            )
        effective_build_path = os.path.join(
            self.build_output_dir,
            "effective_build.yaml",
        )
        try:
            # Record the fully resolved build plan before compilation, including
            # discovered and skipped components, inherited and local settings,
            # inferred types, compiler-facing prefill lengths, and artifact paths.
            with open(effective_build_path, "w", encoding="utf-8") as file:
                yaml.safe_dump(
                    {"build": {"components": resolved_components}},
                    file,
                    sort_keys=False,
                )
        except OSError as exc:
            logger.fatal(
                f"Failed to save effective build configuration "
                f"{effective_build_path}: {exc}"
            )
        logger.info(f"Effective build configuration saved: {effective_build_path!r}")
        results = {}
        for component_name, build_kwargs in build_calls.items():
            logger.info(
                f"Build component started: name={component_name!r}, "
                f"source={build_kwargs['hmonnx']!r}"
            )
            hmm_path = Xh2Exec.build_from_hmonnx(**build_kwargs)
            if hmm_path is None:
                logger.fatal(
                    f"Failed to build component {component_name!r} from "
                    f"{build_kwargs['hmonnx']}"
                )
            results[component_name] = hmm_path
            logger.info(
                f"Build component completed: name={component_name!r}, "
                f"artifact={hmm_path!r}"
            )
        logger.info(
            f"Build stage completed: artifacts={results!r}, "
            f"skipped={skipped_components!r}"
        )
        return results


def lm_main(args, cfg):
    command = args.command
    runner = XH2LmRunner(cfg)
    logger.info(
        f"Large-model task selected: command={command!r}, "
        f"target={runner.target!r}, model={runner.model_name!r}, "
        f"size={runner.model_size!r}, model_type={runner.model_type!r}, "
        f"output={runner.save_dir!r}"
    )
    if command == "quant":
        runner.quant()
    elif command == "build":
        runner.build()
    else:
        logger.fatal(f"Unknown command: {command!r}")
