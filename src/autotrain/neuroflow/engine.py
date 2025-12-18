from __future__ import annotations

import json
import os
import subprocess
import tempfile
import time
from typing import Any, Dict, List, Optional

import psutil
import socket
from autotrain import logger
from autotrain.neuroflow.schemas import NodeKind, PipelineGraph


class GraphExecutor:
    def __init__(self, graph: PipelineGraph, project_name: str = "inference"):
        self.graph = graph
        self.project_name = project_name
        self.node_outputs: Dict[str, Any] = {}
        self.node_map = {node.id: node for node in graph.nodes}

    def prune_ui_processes(self, port: int):
        """Kill any existing process on the target port, but never ourselves."""
        current_pid = os.getpid()
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                if proc.pid == current_pid:
                    continue
                # Check for port in connections
                for conn in proc.connections(kind='inet'):
                    if conn.laddr.port == port:
                        logger.info(f"Killing process {proc.pid} on port {port}")
                        proc.terminate()
                        proc.wait(timeout=3)
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.TimeoutExpired):
                continue

    def get_available_port(self, start_port: int) -> int:
        port = start_port
        while port < start_port + 100:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                if s.connect_ex(('localhost', port)) != 0:
                    return port
            port += 1
        return start_port

    def get_roots(self) -> List[str]:
        targets = {edge.target for edge in self.graph.edges}
        return [node.id for node in self.graph.nodes if node.id not in targets]

    def execute(self, start_node_id: Optional[str] = None, input_data: Any = None):
        """
        Execute the graph starting from a specific node or all roots.
        """
        self.node_outputs = {}  # Clear previous execution results
        if start_node_id:
            return self._execute_node(start_node_id, input_data=input_data)
        
        logger.info(f"[{self.project_name}] Starting graph execution...")
        roots = self.get_roots()
        
        results = {}
        for root_id in roots:
            results[root_id] = self._execute_node(root_id, input_data=input_data)
        return results

    def _execute_node(self, node_id: str, input_data: Any = None):
        if node_id in self.node_outputs and input_data is None:
            return self.node_outputs[node_id]

        node = self.node_map.get(node_id)
        if not node:
            return None

        # if input_data is provided explicitly (for root/start nodes), use it.
        # Otherwise, gather inputs from incoming edges
        if input_data is not None:
            inputs = [input_data]
        else:
            inputs = []
            for edge in self.graph.edges:
                if edge.target == node_id:
                    inputs.append(self._execute_node(edge.source))

        # Combined input (usually just the first one for simple chains)
        combined_input = inputs[0] if inputs else None

        # Execute node logic
        output = self._run_node_logic(node, combined_input)
        self.node_outputs[node_id] = output

        # If it's a router, we follow specific paths based on output
        if node.is_kind(NodeKind.ROUTER):
            # Router logic handles its own next steps
            return output

        # Otherwise, continue to next nodes
        next_edges = [edge for edge in self.graph.edges if edge.source == node_id]
        for edge in next_edges:
            self._execute_node(edge.target)

        return output

    def _run_node_logic(self, node, input_data: Any) -> Any:
        logger.info(f"[{self.project_name}] Running logic for node {node.id} ({node.data.type})")
        config = node.data.config
        
        if node.is_kind(NodeKind.PROMPT):
            template = config.get("template", "")
            # Support multiple variables via input dictionary or simple {{input}}
            if isinstance(input_data, dict):
                res = template
                for k, v in input_data.items():
                    res = res.replace("{{" + k + "}}", str(v))
                return res
            return template.replace("{{input}}", str(input_data or ""))

        if node.is_kind(NodeKind.INFERENCE):
            import requests
            model = config.get("model") or "gpt2"
            provider = config.get("provider", "openai")
            api_key = config.get("api_key") or os.environ.get("HF_TOKEN")
            
            if provider == "huggingface_api":
                API_URL = f"https://api-inference.huggingface.co/models/{model}"
                headers = {"Authorization": f"Bearer {api_key}"}
                payload = {"inputs": str(input_data)}
                response = requests.post(API_URL, headers=headers, json=payload)
                if response.status_code == 200:
                    data = response.json()
                    if isinstance(data, list) and len(data) > 0:
                        return data[0].get("generated_text") or data[0].get("text") or str(data[0])
                    if isinstance(data, dict):
                        return data.get("generated_text") or data.get("text") or str(data)
                    return str(data)
                return f"Error: {response.text}"
            
            if provider == "openai":
                base_url = "https://api.openai.com/v1/chat/completions"
                headers = {
                    "Authorization": f"Bearer {api_key or os.environ.get('OPENAI_API_KEY')}",
                    "Content-Type": "application/json"
                }
                payload = {
                    "model": model if model != "gpt2" else "gpt-3.5-turbo",
                    "messages": [{"role": "user", "content": str(input_data)}]
                }
                response = requests.post(base_url, headers=headers, json=payload)
                if response.status_code == 200:
                    return response.json()["choices"][0]["message"]["content"]
                return f"OpenAI Error: {response.text}"
            
            return f"Inference result (provider {provider}) for: {input_data}"

        if node.is_kind(NodeKind.PDF_LOADER):
            import pypdf
            path = config.get("url") or input_data
            if not path:
                return "No PDF path provided."
            try:
                reader = pypdf.PdfReader(path)
                text = ""
                for page in reader.pages:
                    text += page.extract_text() + "\n"
                return text
            except Exception as e:
                return f"PDF Error: {str(e)}"

        if node.is_kind(NodeKind.TEXT_SPLITTER):
            chunk_size = int(config.get("chunk_size", 1000))
            overlap = int(config.get("chunk_overlap", 200))
            text = str(input_data or "")
            chunks = []
            for i in range(0, len(text), chunk_size - overlap):
                chunks.append(text[i:i + chunk_size])
            return chunks

        if node.is_kind(NodeKind.WEBHOOK):
            return input_data or config.get("payload") or "Webhook Triggered"

        if node.is_kind(NodeKind.ROUTER):
            match_val = config.get("match_value", "")
            cond_type = config.get("condition_type", "contains")
            is_match = False
            str_input = str(input_data).lower()
            str_match = str(match_val).lower()
            if cond_type == "contains": is_match = str_match in str_input
            elif cond_type == "equals": is_match = str_match == str_input
            return {"match": is_match, "data": input_data}

        if node.is_kind(NodeKind.STREAMLIT):
            requested_port = config.get("port") or 8501
            if os.environ.get("NF_IN_UI") == "true":
                return f"http://localhost:{requested_port}"

            self.prune_ui_processes(requested_port)
            port = self.get_available_port(requested_port)
            
            graph_json = self.graph.model_dump_json()
            is_chat = any(n.is_kind(NodeKind.INFERENCE) for n in self.graph.nodes)
            cwd = os.getcwd()
            
            app_code = f"""
import streamlit as st
import json
import os
import sys

# Ensure local autotrain package is importable and prioritized
project_root = {repr(cwd)}
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from autotrain.neuroflow.engine import GraphExecutor
from autotrain.neuroflow.schemas import PipelineGraph

st.set_page_config(page_title="NeuroFlow Interactive App", layout="wide")
st.title("🚀 NeuroFlow Production App")

# Load the graph
graph_json = {repr(graph_json)}
graph = PipelineGraph(**json.loads(graph_json))
executor = GraphExecutor(graph)

if {is_chat}:
    st.subheader("Interactive Chat Interface")
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask something..."):
        st.session_state.messages.append({{"role": "user", "content": prompt}})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            # Run the whole graph starting from roots with this input
            results = executor.execute(input_data=prompt)
            
            # Smart Response Detection: Find the parents of THIS Streamlit node
            this_node_id = {repr(node.id)}
            parents = [e.source for e in executor.graph.edges if e.target == this_node_id]
            
            if parents:
                response = executor.node_outputs.get(parents[0], "No output from node: " + parents[0])
            else:
                # Heuristic fallback: Find any INFERENCE node output
                inference_nodes = [n.id for n in executor.graph.nodes if n.is_kind(NodeKind.INFERENCE)]
                if inference_nodes:
                    response = executor.node_outputs.get(inference_nodes[0], "Inference logic did not execute.")
                else:
                    # Final fallback: Last non-UI output from terminal nodes
                    response = list(results.values())[-1] if results else "Graph executed but no output found."
            
            st.markdown(str(response))
            st.session_state.messages.append({{"role": "assistant", "content": str(response)}})
else:
    st.subheader("Interactive Pipeline Preview")
    roots = executor.get_roots()
    inputs = {{}}
    with st.form("pipeline_input"):
        for root_id in roots:
            r_node = executor.node_map[root_id]
            inputs[root_id] = st.text_input(f"Input for {{r_node.data.type}} ({{root_id}})")
        
        submitted = st.form_submit_button("Run Pipeline")
        if submitted:
            main_input = list(inputs.values())[0] if inputs else ""
            results = executor.execute(input_data=main_input)
            st.write("### Outputs")
            st.json(results)

st.divider()
with st.expander("System Metadata"):
    st.write("Node ID: {node.id}")
    st.json({repr(config)})
"""
            temp_dir = tempfile.gettempdir()
            app_path = os.path.join(temp_dir, f"nf_streamlit_{node.id}.py")
            with open(app_path, "w", encoding="utf-8") as f:
                f.write(app_code)
            
            # Use streamlit run in a subprocess
            cmd = ["streamlit", "run", app_path, "--server.port", str(port), "--server.headless", "true"]
            try:
                log_path = os.path.join(temp_dir, f"nf_streamlit_{node.id}.log")
                env = os.environ.copy()
                env["NF_IN_UI"] = "true"
                with open(log_path, "w") as log_file:
                    subprocess.Popen(cmd, stdout=log_file, stderr=log_file, start_new_session=True, env=env)
                logger.info(f"Streamlit launched in background. Logs: {log_path}")
                time.sleep(2)
                return f"http://localhost:{port}"
            except Exception as e:
                return f"Streamlit Launch Error: {str(e)}"

        if node.is_kind(NodeKind.GRADIO):
            requested_port = config.get("port") or 7860
            if os.environ.get("NF_IN_UI") == "true":
                return f"http://localhost:{requested_port}"

            self.prune_ui_processes(requested_port)
            port = self.get_available_port(requested_port)
            
            graph_json = self.graph.model_dump_json()
            is_chat = any(n.is_kind(NodeKind.INFERENCE) for n in self.graph.nodes)
            cwd = os.getcwd()
            
            app_code = f"""
import gradio as gr
import json
import os
import sys

# Ensure local autotrain package is importable and prioritized
project_root = {repr(cwd)}
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from autotrain.neuroflow.engine import GraphExecutor
from autotrain.neuroflow.schemas import PipelineGraph

# Load the graph
graph_json = {repr(graph_json)}
graph = PipelineGraph(**json.loads(graph_json))
executor = GraphExecutor(graph)

def run_pipeline(user_input, history=None):
    results = executor.execute(input_data=user_input)
    
    # Smart Response Detection: Find the parents of the GRADIO node
    this_node_id = {repr(node.id)}
    parents = [e.source for e in executor.graph.edges if e.target == this_node_id]
    
    if parents:
        response = executor.node_outputs.get(parents[0])
    else:
        # Fallback to inference or last terminal
        inference_nodes = [n.id for n in executor.graph.nodes if n.is_kind(NodeKind.INFERENCE)]
        if inference_nodes:
            response = executor.node_outputs.get(inference_nodes[0])
        else:
            outputs = list(results.values())
            response = outputs[-1] if outputs else str(results)
    
    return str(response)

if {is_chat}:
    demo = gr.ChatInterface(
        fn=run_pipeline,
        title="🚀 NeuroFlow Chat Production App",
        description="This interface was automatically generated from your NeuroFlow graph."
    )
else:
    roots = executor.get_roots()
    def form_wrapper(*inputs_list):
        # Taking the first input for simplicity
        user_input = inputs_list[0] if inputs_list else ""
        return run_pipeline(user_input)

    with gr.Blocks(title="NeuroFlow Production App") as demo:
        gr.Markdown("# 🚀 NeuroFlow Production App")
        with gr.Column():
            input_widgets = []
            for root_id in roots:
                r_node = executor.node_map[root_id]
                input_widgets.append(gr.Textbox(label=f"Input for {{r_node.data.type}} ({{root_id}})"))
            
            run_btn = gr.Button("Run Pipeline")
            output = gr.Textbox(label="Output")
            
            run_btn.click(fn=form_wrapper, inputs=input_widgets, outputs=output)

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port={port}, prevent_thread_lock=True)
"""
            temp_dir = tempfile.gettempdir()
            app_path = os.path.join(temp_dir, f"nf_gradio_{node.id}.py")
            with open(app_path, "w", encoding="utf-8") as f:
                f.write(app_code)
            
            cmd = ["python", app_path]
            try:
                log_path = os.path.join(temp_dir, f"nf_gradio_{node.id}.log")
                env = os.environ.copy()
                env["NF_IN_UI"] = "true"
                with open(log_path, "w") as log_file:
                    subprocess.Popen(cmd, stdout=log_file, stderr=log_file, start_new_session=True, env=env)
                logger.info(f"Gradio launched in background. Logs: {log_path}")
                time.sleep(2)
                return f"http://localhost:{port}"
            except Exception as e:
                return f"Gradio Launch Error: {str(e)}"

        if node.is_kind(NodeKind.JAVASCRIPT):
            # Placeholder for JS execution (requires external engine like node or py-mini-racer)
            return f"JS Execution Placeholder: {config.get('code')}"

        return input_data
