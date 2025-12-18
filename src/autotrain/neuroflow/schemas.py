from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, ConfigDict


class NodeKind(str, Enum):
    DATASET = "dataset"
    MODEL = "model"
    TASK = "task"
    TRAINER = "trainer"
    HARDWARE = "hardware"
    DEPLOY = "deploy"
    GRADIO = "gradio_ui"
    STREAMLIT = "streamlit_ui"
    PROMPT = "prompt"
    INFERENCE = "inference"
    ROUTER = "router"
    WEBHOOK = "webhook"
    JAVASCRIPT = "javascript"
    PYTHON_REPL = "python_repl"
    GOOGLE_SEARCH = "google_search"
    API_CALL = "api_call"
    EMAIL = "email"
    PDF_LOADER = "pdf_loader"
    TEXT_SPLITTER = "text_splitter"
    EMBEDDINGS = "embeddings"
    VECTOR_DB = "vector_db"
    TOOL = "tool"


class NodePayload(BaseModel):
    label: str
    type: str
    config: Dict[str, Any] = Field(default_factory=dict)


class GraphNode(BaseModel):
    id: str
    type: str
    data: NodePayload

    def is_kind(self, kind: NodeKind) -> bool:
        return self.data.type == kind.value


class GraphEdge(BaseModel):
    id: str
    source: str
    target: str


class PipelineGraph(BaseModel):
    nodes: List[GraphNode]
    edges: List[GraphEdge] = Field(default_factory=list)
    username_hint: Optional[str] = Field(default=None, alias="username")
    model_config = ConfigDict(populate_by_name=True)

    def find_first(self, kind: NodeKind) -> Optional[GraphNode]:
        for node in self.nodes:
            if node.is_kind(kind):
                return node
        return None

    def require(self, kind: NodeKind) -> GraphNode:
        node = self.find_first(kind)
        if node is None:
            from autotrain.neuroflow.exceptions import NeuroFlowValidationError

            raise NeuroFlowValidationError(f"Missing required `{kind.value}` node.")
        return node


class CompileResponse(BaseModel):
    config: Dict[str, Any]
    cli_command: str
    python_code: str
    project_name: str
    task: str
    base_model: str
    backend: str
    hub_repo: Optional[str] = None
    job_id: Optional[str] = None
    outputs: Optional[Dict[str, Any]] = None
