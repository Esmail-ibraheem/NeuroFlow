from __future__ import annotations

import json
import os
import uuid
from dataclasses import dataclass
from typing import Optional

from autotrain import logger
from autotrain.app.params import AppParams
from autotrain.app.utils import token_verification
from autotrain.neuroflow.compiler import CompilationArtifacts, compile_pipeline
from autotrain.neuroflow.exceptions import NeuroFlowConfigError
from autotrain.neuroflow.schemas import PipelineGraph
from autotrain.project import AutoTrainProject


@dataclass
class RunResult:
    job_id: str
    compilation: CompilationArtifacts
    outputs: Optional[Dict[str, Any]] = None


def run_pipeline(graph: PipelineGraph, token: Optional[str], username_hint: Optional[str] = None) -> RunResult:
    """
    Compile a NeuroFlow graph and trigger the corresponding AutoTrain backend job.
    """
    artifacts = compile_pipeline(graph)
    backend = artifacts.backend

    if artifacts.pipeline_mode == "production":
        from autotrain.neuroflow.engine import GraphExecutor
        logger.info("Executing Production pipeline...")
        executor = GraphExecutor(graph, project_name=artifacts.project_name)
        results = executor.execute()
        # For production mode, we return a success status since there's no backend job ID
        return RunResult(job_id=f"production-{uuid.uuid4().hex[:8]}", compilation=artifacts, outputs=executor.node_outputs)

    effective_token = token or os.environ.get("HF_TOKEN")
    backend_requires_token = not backend.startswith("local")
    if backend_requires_token and not effective_token:
        raise NeuroFlowConfigError(
            f"A Hugging Face token is required for backend `{backend}`. "
            "Please add a Deploy node with a valid token."
        )
    if effective_token is None:
        effective_token = ""

    username = _resolve_username(effective_token, username_hint or artifacts.username, backend=backend)

    app_params = AppParams(
        job_params_json=json.dumps(artifacts.params_payload),
        token=effective_token,
        project_name=artifacts.project_name,
        username=username,
        task=artifacts.autotrain_task,
        data_path=artifacts.dataset.path,
        base_model=artifacts.base_model,
        column_mapping=artifacts.column_mapping,
        train_split=artifacts.dataset.split,
        valid_split=artifacts.dataset.valid_split,
        using_hub_dataset=artifacts.using_hub_dataset,
        api=True,
        revision=artifacts.revision,
    )

    params = app_params.munge()
    project = AutoTrainProject(params=params, backend=backend)
    job_id = project.create()

    if artifacts.pipeline_mode == "hybrid":
        from autotrain.neuroflow.engine import GraphExecutor
        logger.info("Hybrid mode: Triggered training job %s. Also booting production engine...", job_id)
        executor = GraphExecutor(graph, project_name=artifacts.project_name)
        # Note: In hybrid mode we might want to run in background or wait, 
        # but for now we'll just start it.
        executor.execute()
        production_outputs = executor.node_outputs
    else:
        production_outputs = None

    logger.info("NeuroFlow pipeline triggered backend job %s", job_id)
    return RunResult(job_id=str(job_id), compilation=artifacts, outputs=production_outputs)


def _resolve_username(token: Optional[str], username_hint: Optional[str], backend: str) -> str:
    if backend.startswith("local"):
        return username_hint or "local-user"
    if username_hint:
        return username_hint
    if token:
        logger.info("Fetching username from token for backend %s", backend)
        info = token_verification(token=token)
        username = info.get("name")
        if username:
            return username
    raise NeuroFlowConfigError("A Hugging Face username is required for remote backends.")
