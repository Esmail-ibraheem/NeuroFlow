import json
from autotrain.neuroflow.compiler import compile_pipeline
from autotrain.neuroflow.schemas import PipelineGraph, GraphNode, GraphEdge, NodePayload

def test_compilation():
    # 1. Create a mock graph
    nodes = [
        GraphNode(id="dataset_0", type="dataset", data=NodePayload(label="Dataset", type="dataset", config={
            "path": "imdb ", # Trailing space to test stripping
            "split": "train",
            "train_test_split": 0.1,
            "text_column": "text"
        })),
        GraphNode(id="model_0", type="model", data=NodePayload(label="Model", type="model", config={
            "modelId": "gpt2",
            "revision": "main",
            "quantization": None
        })),
        GraphNode(id="task_0", type="task", data=NodePayload(label="Task", type="task", config={
            "task": "llm-sft"
        })),
        GraphNode(id="trainer_0", type="trainer", data=NodePayload(label="Trainer", type="trainer", config={
            "project_name": "test-project",
            "epochs": 1,
            "batch_size": 4
        })),
        GraphNode(id="hardware_0", type="hardware", data=NodePayload(label="Hardware", type="hardware", config={
            "backend": "Hugging Face Spaces",
            "accelerator": "A10G (Medium)",
            "numGpus": 2
        }))
    ]
    edges = [
        GraphEdge(id="e1", source="dataset_0", target="task_0"),
        GraphEdge(id="e2", source="model_0", target="task_0"),
        GraphEdge(id="e3", source="task_0", target="trainer_0"),
        GraphEdge(id="e4", source="trainer_0", target="hardware_0")
    ]
    graph = PipelineGraph(nodes=nodes, edges=edges)

    # 2. Compile
    artifacts = compile_pipeline(graph)

    # 3. Assertions
    print(f"Project Name: {artifacts.project_name}")
    print(f"Backend: {artifacts.backend}")
    print(f"Dataset Path: '{artifacts.dataset.path}'")
    print(f"Train/Test Split: {artifacts.dataset.train_test_split}")
    print(f"Revision: {artifacts.revision}")
    print(f"Num GPUs in Payload: {artifacts.params_payload.get('num_gpus')}")

    assert artifacts.dataset.path == "imdb"
    assert artifacts.dataset.train_test_split == 0.1
    assert artifacts.revision == "main"
    assert artifacts.params_payload.get("num_gpus") == 2
    assert artifacts.params_payload.get("revision") == "main"
    assert artifacts.params_payload.get("quantization") is None

    print("\nCompilation Test Passed!")

if __name__ == "__main__":
    test_compilation()
