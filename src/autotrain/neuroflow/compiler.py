from __future__ import annotations

import copy
import json
import re
import uuid
from dataclasses import dataclass
from typing import Any, Dict, Optional

from autotrain import logger
from autotrain.app.params import PARAMS
from autotrain.neuroflow.exceptions import NeuroFlowConfigError, NeuroFlowValidationError
from autotrain.neuroflow.schemas import NodeKind, PipelineGraph


@dataclass
class DatasetSpec:
    path: str
    split: str
    valid_split: Optional[str] = None
    train_test_split: Optional[float] = None


@dataclass
class CompilationArtifacts:
    config: Dict[str, Any]
    cli_command: str
    python_code: str
    project_name: str
    autotrain_task: str
    base_model: str
    backend: str
    dataset: DatasetSpec
    column_mapping: Dict[str, Any]
    params_payload: Dict[str, Any]
    username: Optional[str] = None
    hub_repo: Optional[str] = None
    revision: Optional[str] = None
    hub_visibility: str = "private"
    using_hub_dataset: bool = True
    pipeline_mode: str = "training"


TASK_MAPPING = {
    "llm-sft": "llm:sft",
    "llm-dpo": "llm:dpo",
    "llm-orpo": "llm:orpo",
    "llm-generic": "llm:generic",
    "llm-reward": "llm:reward",
    "st-pair": "st:pair",
    "st-pair-classification": "st:pair_class",
    "st-pair-scoring": "st:pair_score",
    "st-triplet": "st:triplet",
    "st-question-answering": "st:qa",
    "text-classification": "text-classification",
    "text-regression": "text-regression",
    "extractive-question-answering": "extractive-question-answering",
    "sequence-to-sequence": "seq2seq",
    "token-classification": "token-classification",
    "sentence-transformers": "st:pair",
    "image-classification": "image-classification",
    "image-regression": "image-regression",
    "object-detection": "image-object-detection",
    "tabular-classification": "tabular-classification",
    "tabular-regression": "tabular-regression",
    "vlm-captioning": "vlm:captioning",
    "vlm-vqa": "vlm:vqa",
    "dreambooth": "dreambooth",
}

HARDWARE_DEFAULT = "local-ui"

TRAINER_PARAM_ALIASES = {
    "learning_rate": "lr",
    "lr": "lr",
    "num_train_epochs": "epochs",
    "epochs": "epochs",
    "batch_size": "batch_size",
    "block_size": "block_size",
    "optimizer": "optimizer",
    "scheduler": "scheduler",
    "mixed_precision": "mixed_precision",
    "gradient_accumulation": "gradient_accumulation",
    "warmup_ratio": "warmup_ratio",
    "weight_decay": "weight_decay",
    "use_peft": "peft",
    "lora_r": "lora_r",
    "lora_alpha": "lora_alpha",
    "lora_dropout": "lora_dropout",
    "target_modules": "target_modules",
    "chat_template": "chat_template",
    "quantization": "quantization",
    "gradient_checkpointing": "disable_gradient_checkpointing",
}

FIELD_CASTS = {
    "lr": float,
    "epochs": int,
    "batch_size": int,
    "block_size": int,
    "gradient_accumulation": int,
    "warmup_ratio": float,
    "weight_decay": float,
    "lora_r": int,
    "lora_alpha": int,
    "lora_dropout": float,
}


def compile_pipeline(graph: PipelineGraph) -> CompilationArtifacts:
    trainer_node = graph.find_first(NodeKind.TRAINER)
    is_training_required = trainer_node is not None

    if is_training_required:
        dataset_node = graph.require(NodeKind.DATASET)
        model_node = graph.require(NodeKind.MODEL)
        task_node = graph.require(NodeKind.TASK)
    else:
        dataset_node = graph.find_first(NodeKind.DATASET)
        model_node = graph.find_first(NodeKind.MODEL)
        task_node = graph.find_first(NodeKind.TASK)

    hardware_node = graph.find_first(NodeKind.HARDWARE)
    deploy_node = graph.find_first(NodeKind.DEPLOY)
    gradio_node = graph.find_first(NodeKind.GRADIO)
    streamlit_node = graph.find_first(NodeKind.STREAMLIT)

    task_value = task_node.data.config.get("task", "llm-sft") if task_node else "llm-generic"
    autotrain_task = _map_task(task_value)
    base_model = model_node.data.config.get("modelId") if model_node else "external-api"
    revision = model_node.data.config.get("revision") if model_node else None

    dataset_spec = _build_dataset_spec(dataset_node) if dataset_node else DatasetSpec(path="None", split="train")
    using_hub_dataset = dataset_node.data.config.get("source_type", "hub") == "hub" if dataset_node else False
    backend_choice = _resolve_backend(hardware_node)
    
    # Defaults in case trainer_node is missing
    trainer_config = trainer_node.data.config if trainer_node else {}
    model_config = model_node.data.config if model_node else {}
    params_payload = _build_params_payload(
        autotrain_task, trainer_config, model_config, deploy_node, hardware_node
    )
    if revision:
        params_payload["revision"] = revision

    slug_seed = dataset_node.data.config.get("path", "neuroflow") if dataset_node else "inference"
    label_seed = deploy_node.data.config.get("hubId") if deploy_node else slug_seed
    project_name = _slugify(label_seed or slug_seed or "neuroflow")
    if not project_name:
        project_name = f"neuroflow-{uuid.uuid4().hex[:8]}"

    username, hub_repo, hub_visibility = _resolve_hub_metadata(deploy_node, project_name)

    is_training = trainer_node is not None
    production_kinds = {
        NodeKind.INFERENCE, NodeKind.PROMPT, NodeKind.WEBHOOK,
        NodeKind.ROUTER, NodeKind.GRADIO, NodeKind.STREAMLIT,
        NodeKind.PDF_LOADER, NodeKind.VECTOR_DB, NodeKind.API_CALL,
        NodeKind.TOOL, NodeKind.JAVASCRIPT, NodeKind.PYTHON_REPL
    }
    is_production = any(node.data.type in [k.value for k in production_kinds] for node in graph.nodes)

    pipeline_mode = "training"
    if is_training and is_production:
        pipeline_mode = "hybrid"
    elif is_production:
        pipeline_mode = "production"

    config = {
        "task": autotrain_task,
        "base_model": base_model,
        "project_name": project_name,
        "backend": backend_choice,
        "pipeline_mode": pipeline_mode,
        "data": {
            "path": dataset_spec.path,
            "train_split": dataset_spec.split,
            "valid_split": dataset_spec.valid_split,
            "train_test_split": dataset_spec.train_test_split,
            "column_mapping": _build_column_mapping(autotrain_task, dataset_node.data.config) if dataset_node else {},
            "using_hub_dataset": using_hub_dataset,
        },
        "params": params_payload,
    }
    if revision:
        config["data"]["revision"] = revision

    if gradio_node:
        config["gradio_ui"] = gradio_node.data.config
    if streamlit_node:
        config["streamlit_ui"] = streamlit_node.data.config
    if hub_repo:
        config["hub"] = {"repo_id": hub_repo, "visibility": hub_visibility}

    cli_command = _render_cli_command()
    python_code = _render_python_code(
        project_name=project_name,
        autotrain_task=autotrain_task,
        backend=backend_choice,
        dataset_spec=dataset_spec,
        base_model=base_model,
        params_payload=params_payload,
        column_mapping=_build_column_mapping(autotrain_task, dataset_node.data.config) if dataset_node else {},
        username=username,
        pipeline_mode=pipeline_mode,
        graph_json=graph.model_dump_json(indent=4),
    )

    return CompilationArtifacts(
        config=config,
        cli_command=cli_command,
        python_code=python_code,
        autotrain_task=autotrain_task,
        project_name=project_name,
        base_model=base_model,
        dataset=dataset_spec,
        params_payload=params_payload,
        backend=backend_choice,
        column_mapping=_build_column_mapping(autotrain_task, dataset_node.data.config) if dataset_node else {},
        username=username,
        hub_repo=hub_repo,
        revision=revision,
        hub_visibility=hub_visibility,
        using_hub_dataset=using_hub_dataset,
        pipeline_mode=pipeline_mode,
    )


def _map_task(task_value: str) -> str:
    mapped = TASK_MAPPING.get(task_value)
    if not mapped:
        raise NeuroFlowValidationError(f"Task `{task_value}` is not supported yet.")
    return mapped


def _build_dataset_spec(node: GraphNode) -> DatasetSpec:
    cfg = node.data.config
    path = cfg.get("path") or ""
    split = cfg.get("split") or "train"
    valid_split = cfg.get("valid_split")
    train_test_split = cfg.get("train_test_split")

    if not valid_split and not train_test_split:
        valid_split = "validation"

    return DatasetSpec(
        path=path.strip(),
        split=split.strip(),
        valid_split=valid_split.strip() if valid_split else None,
        train_test_split=train_test_split,
    )


def _build_column_mapping(task: str, cfg: Dict[str, Any]) -> Dict[str, Any]:
    def require(field: str) -> str:
        value = cfg.get(field)
        if not value:
            raise NeuroFlowConfigError(f"Dataset column `{field}` is required for task `{task}`.")
        return value

    mapping: Dict[str, Any] = {}
    if task.startswith("llm:"):
        mapping["text_column"] = require("text_column")
        prompt_col = cfg.get("prompt_column") or cfg.get("prompt_text_column")
        rejected_col = cfg.get("rejected_text_column")
        if task in ("llm:dpo", "llm:orpo", "llm:reward"):
            if not prompt_col:
                raise NeuroFlowConfigError("Prompt column is required for the selected LLM trainer.")
            if not rejected_col:
                raise NeuroFlowConfigError("Rejected column is required for the selected LLM trainer.")
        mapping["prompt_text_column"] = prompt_col or cfg.get("prompt_text_column")
        mapping["rejected_text_column"] = rejected_col
    elif task == "text-classification" or task == "text-regression":
        mapping["text_column"] = require("text_column")
        mapping["target_column"] = require("target_column")
    elif task == "seq2seq":
        mapping["text_column"] = require("text_column")
        mapping["target_column"] = require("target_column")
    elif task == "extractive-question-answering":
        mapping["text_column"] = require("text_column")
        mapping["question_column"] = require("question_column")
        mapping["answer_column"] = require("answer_column")
    elif task == "token-classification":
        mapping["tokens_column"] = require("tokens_column")
        mapping["tags_column"] = require("tags_column")
    elif task.startswith("st:"):
        mapping["sentence1_column"] = require("sentence1_column")
        mapping["sentence2_column"] = require("sentence2_column")
        if task == "st:triplet":
            mapping["sentence3_column"] = require("sentence3_column")
        if task in ("st:pair_class", "st:pair_score"):
            mapping["target_column"] = require("target_column")
    elif task == "image-classification" or task == "image-regression":
        mapping["image_column"] = require("image_column")
        mapping["target_column"] = require("target_column")
    elif task == "image-object-detection":
        mapping["image_column"] = require("image_column")
        mapping["objects_column"] = require("objects_column")
    elif task.startswith("tabular"):
        mapping["id_column"] = require("id_column")
        labels = cfg.get("target_columns") or cfg.get("target_column") or cfg.get("label")
        if isinstance(labels, str):
            labels = [lbl.strip() for lbl in labels.split(",") if lbl.strip()]
        mapping["target_columns"] = labels or []
    elif task.startswith("vlm:"):
        mapping["image_column"] = require("image_column")
        mapping["text_column"] = require("text_column")
        mapping["prompt_text_column"] = require("prompt_column")
    else:
        logger.warning("No explicit column mapping for task %s; using defaults.", task)
    return mapping


def _build_params_payload(
    task: str,
    trainer_cfg: Dict[str, Any],
    model_cfg: Dict[str, Any],
    deploy_node,
    hardware_node,
) -> Dict[str, Any]:
    base_key = _resolve_param_key(task)
    base_params = copy.deepcopy(PARAMS.get(base_key, {}))
    if task.startswith("llm:"):
        base_params["trainer"] = task.split(":")[1]
    if task.startswith("st:"):
        base_params["trainer"] = task.split(":")[1]
    if task.startswith("vlm:"):
        base_params["trainer"] = task.split(":")[1]

    merged = base_params
    for user_key, user_value in trainer_cfg.items():
        target_key = TRAINER_PARAM_ALIASES.get(user_key)
        if not target_key or user_value == "":
            continue
        cast_fn = FIELD_CASTS.get(target_key)
        value = user_value
        try:
            if cast_fn:
                value = cast_fn(user_value)
        except (TypeError, ValueError):
            logger.warning("Could not cast trainer field %s with value %s", user_key, user_value)

        if target_key == "disable_gradient_checkpointing":
            merged[target_key] = not bool(user_value)
        else:
            merged[target_key] = value

    # Merge relevant model_cfg parameters
    for user_key, user_value in model_cfg.items():
        target_key = TRAINER_PARAM_ALIASES.get(user_key)
        if not target_key or user_value == "":
            continue
        
        # Don't overwrite trainer_cfg values with model_cfg values if already set
        if target_key in merged and user_key in trainer_cfg:
            continue
            
        merged[target_key] = user_value

    if "peft" not in merged:
        merged["peft"] = True
    if "quantization" not in merged:
        merged["quantization"] = model_cfg.get("quantization")

    quantization_value = merged.get("quantization")
    if isinstance(quantization_value, str) and quantization_value.lower() in ("none", "no"):
        merged["quantization"] = None

    if "chat_template" in model_cfg and model_cfg["chat_template"] != "none":
        merged["chat_template"] = model_cfg["chat_template"]
    elif merged.get("chat_template") == "none":
        merged["chat_template"] = None

    if hardware_node:
        num_gpus = hardware_node.data.config.get("numGpus")
        if num_gpus:
            merged["num_gpus"] = int(num_gpus)

    return merged


def _resolve_param_key(task: str) -> str:
    if task.startswith("llm:"):
        return "llm"
    if task.startswith("st:"):
        return "st"
    if task.startswith("vlm:"):
        return "vlm"
    if task.startswith("tabular"):
        return "tabular"
    if task == "image-object-detection":
        return "image-object-detection"
    if task == "extractive-question-answering":
        return "extractive-qa"
    return task


def _resolve_backend(node: Optional[Any]) -> str:
    if not node:
        return HARDWARE_DEFAULT
    backend_choice = node.data.config.get("backend")
    accelerator = node.data.config.get("accelerator")
    if not backend_choice:
        return HARDWARE_DEFAULT

    backend_choice = backend_choice.lower()
    if backend_choice.startswith("local"):
        return "local-ui"
    if "spaces" in backend_choice:
        return {
            "t4 (small)": "spaces-t4-small",
            "a10g (medium)": "spaces-a10g-small",
            "a100 (large)": "spaces-a100-large",
            "h100 (massive)": "spaces-a10g-largex4",
        }.get((accelerator or "").lower(), "spaces-a10g-small")
    if "dgx" in backend_choice:
        return "dgx-a100"
    if "aws" in backend_choice or "sagemaker" in backend_choice:
        return {
            "t4 (small)": "ep-aws-useast1-s",
            "a10g (medium)": "ep-aws-useast1-m",
            "a100 (large)": "ep-aws-useast1-l",
            "h100 (massive)": "ep-aws-useast1-2xl",
        }.get((accelerator or "").lower(), "ep-aws-useast1-m")
    return HARDWARE_DEFAULT


def _resolve_hub_metadata(node, project_name: str):
    if not node:
        return None, None, "private"
    hub_id = node.data.config.get("hubId") or node.data.config.get("repo_id")
    visibility = "private" if node.data.config.get("private") else "public"
    if not hub_id:
        return None, None, visibility
    hub_id = hub_id.strip()
    if "/" not in hub_id:
        raise NeuroFlowValidationError("Hub repo must include `username/repo_name`.")
    owner, repo = hub_id.split("/", 1)
    repo = repo or f"autotrain-{project_name}"
    return owner, f"{owner}/{repo}", visibility


def _slugify(value: str) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9\\-]+", "-", value)
    normalized = normalized.strip("-").lower()
    if not normalized:
        return normalized
    return normalized[:64]


def _render_cli_command() -> str:
    return (
        "curl -X POST $AUTOTRAIN_API_URL/neuroflow/run \\\n"
        "  -H \"Authorization: Bearer $HF_TOKEN\" \\\n"
        "  -H \"Content-Type: application/json\" \\\n"
        "  --data '@pipeline.json'"
    )


def _render_python_code(
    *,
    project_name: str,
    autotrain_task: str,
    backend: str,
    dataset_spec: DatasetSpec,
    base_model: str,
    params_payload: Dict[str, Any],
    column_mapping: Dict[str, Any],
    username: Optional[str],
    pipeline_mode: str = "training",
    graph_json: str = "{}",
) -> str:
    params_literal = json.dumps(params_payload, indent=4)
    column_literal = json.dumps(column_mapping, indent=4)
    train_split = dataset_spec.split
    valid_split = json.dumps(dataset_spec.valid_split)
    username_literal = username or "<your-hf-username>"
    
    code = f'''import json
import os

from autotrain.app.params import AppParams
from autotrain.project import AutoTrainProject
from autotrain.neuroflow.engine import GraphExecutor
from autotrain.neuroflow.schemas import PipelineGraph

pipeline_mode = "{pipeline_mode}"
graph_data = {graph_json}
graph = PipelineGraph(**graph_data)

params_json = {params_literal}
column_mapping = {column_literal}

def run_training():
    app_params = AppParams(
        job_params_json=json.dumps(params_json),
        token=os.environ.get("HF_TOKEN"),
        project_name="{project_name}",
        username="{username_literal}",
        task="{autotrain_task}",
        data_path="{dataset_spec.path}",
        base_model="{base_model}",
        column_mapping=column_mapping,
        train_split="{train_split}",
        valid_split={valid_split},
        using_hub_dataset=True,
        api=True,
    )

    params = app_params.munge()
    project = AutoTrainProject(params=params, backend="{backend}")
    return project.create()

def run_production():
    executor = GraphExecutor(graph)
    executor.execute()

if pipeline_mode == "training":
    run_training()
elif pipeline_mode == "production":
    run_production()
elif pipeline_mode == "hybrid":
    print("Step 1: Training...")
    job_id = run_training()
    print(f"Training started (Job ID: {{job_id}}). Step 2: Booting Production Engine...")
    # In hybrid mode, we might wait for training to finish or run in parallel
    run_production()
'''
    return code
