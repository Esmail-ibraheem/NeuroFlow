import { Node, Edge } from 'reactflow';
import { NodeData, NodeType, TaskType } from '../types';

interface CompiledPipeline {
  config: Record<string, any>;
  cliCommand: string;
  pythonCode: string;
  validationErrors: string[];
}

export const compilePipeline = (nodes: Node<NodeData>[], edges: Edge[]): CompiledPipeline => {
  const errors: string[] = [];
  const config: Record<string, any> = {};

  // 1. Extract Nodes by Type
  const datasetNode = nodes.find(n => n.data.type === NodeType.DATASET);
  const modelNode = nodes.find(n => n.data.type === NodeType.MODEL);
  const taskNode = nodes.find(n => n.data.type === NodeType.TASK);
  const trainerNode = nodes.find(n => n.data.type === NodeType.TRAINER);
  const hardwareNode = nodes.find(n => n.data.type === NodeType.HARDWARE);
  const deployNode = nodes.find(n => n.data.type === NodeType.DEPLOY);
  const gradioNode = nodes.find(n => n.data.type === NodeType.GRADIO_UI);
  const streamlitNode = nodes.find(n => n.data.type === NodeType.STREAMLIT_UI);

  // 2. Validation
  // If no model or task, we assume this might be just an inference pipeline? 
  // But for now, we enforce training basics if trainer is present.
  if (trainerNode) {
    if (!datasetNode) errors.push("Missing Dataset Node: You need input data.");
    if (!modelNode) errors.push("Missing Model Node: Select a base model.");
    if (!taskNode) errors.push("Missing Task Node: Define what the model should do (e.g., LLM SFT).");
  }

  if (errors.length > 0) {
    return { config: {}, cliCommand: "", pythonCode: "", validationErrors: errors };
  }

  // 3. Build Configuration Object (AutoTrain Schema)
  const task = taskNode?.data.config.task || 'llm-sft';
  const baseModel = modelNode?.data.config.modelId || 'meta-llama/Meta-Llama-3-8B';
  const projectName = `neuroflow-${task}-${Date.now()}`;

  // Base configuration
  config['task'] = task;
  config['base_model'] = baseModel;
  config['project_name'] = projectName;

  // Dataset Config
  if (datasetNode) {
    config['data'] = {
      path: datasetNode.data.config.path || 'imdb',
      type: datasetNode.data.config.source_type || 'hub',
      train_split: datasetNode.data.config.split || 'train',
      valid_split: 'validation',
      column_mapping: {
        text_column: datasetNode.data.config.text_column || 'text',
        target_column: datasetNode.data.config.target_column || 'label'
      }
    };
  }

  // Trainer Config
  if (trainerNode) {
    const tConfig = trainerNode.data.config || {};
    config['params'] = {
      learning_rate: parseFloat(tConfig.learning_rate) || 2e-4,
      num_train_epochs: parseInt(tConfig.num_train_epochs) || 1,
      batch_size: parseInt(tConfig.batch_size) || 2,
      block_size: parseInt(tConfig.block_size) || 1024,
      optimizer: tConfig.optimizer || 'adamw_torch',
      scheduler: tConfig.scheduler || 'linear',
      mixed_precision: tConfig.mixed_precision || 'fp16',

      // PEFT
      peft: tConfig.use_peft !== false,
      lora_r: parseInt(tConfig.lora_r) || 16,
      lora_alpha: parseInt(tConfig.lora_alpha) || 32,
      lora_dropout: parseFloat(tConfig.lora_dropout) || 0.05,
      target_modules: tConfig.target_modules ? tConfig.target_modules.split(',') : ['all-linear']
    };
  }

  // Hardware Config
  if (hardwareNode && deployNode) {
    config['hub'] = {
      username: deployNode.data.config.hubId?.split('/')[0] || 'neuroflow-user',
      token: deployNode.data.config.token || '<HF_TOKEN_HERE>',
      push_to_hub: !!deployNode,
    };
  }

  // 4. Generate Python Code
  let pythonCode = `import os
import torch
`;

  // --- Training Section ---
  if (trainerNode) {
    pythonCode += `from autotrain.params import LLMTrainingParams
from autotrain.trainers.clm import train as train_llm

# NeuroFlow Auto-Generated Training Script
# Task: ${task}
# Base Model: ${baseModel}

def run_training():
    params = LLMTrainingParams(
        model="${baseModel}",
        data_path="${config.data?.path}",
        train_split="${config.data?.train_split}",
        text_column="${config.data?.column_mapping?.text_column}",
        learning_rate=${config.params?.learning_rate},
        num_train_epochs=${config.params?.num_train_epochs},
        train_batch_size=${config.params?.batch_size},
        block_size=${config.params?.block_size},
        lora_r=${config.params?.lora_r},
        lora_alpha=${config.params?.lora_alpha},
        lora_dropout=${config.params?.lora_dropout},
        target_modules=${JSON.stringify(config.params?.target_modules)},
        mixed_precision="${config.params?.mixed_precision}",
        peft=${config.params?.peft ? 'True' : 'False'},
        project_name="${projectName}",
    )

    print(f"Starting training for project: {params.project_name}")
    train_llm(params)

if __name__ == "__main__":
    run_training()
`;
  }

  // --- UI Section (Gradio / Streamlit) ---

  if (gradioNode) {
    const gConfig = gradioNode.data.config;
    pythonCode += `
# ==========================================
# Gradio Inference Interface
# ==========================================
import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer

def launch_gradio_app():
    # Load model (Assuming it's loaded from the training output or base model)
    model_id = "./${projectName}" if os.path.exists("./${projectName}") else "${baseModel}"
    
    print(f"Loading model for inference: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, device_map="auto")

    def generate(prompt):
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        outputs = model.generate(**inputs, max_new_tokens=200)
        return tokenizer.decode(outputs[0], skip_special_tokens=True)

    demo = gr.Interface(
        fn=generate,
        inputs="${gConfig.input_type || 'text'}",
        outputs="${gConfig.output_type || 'text'}",
        title="${gConfig.title || 'NeuroFlow AI App'}",
        theme="${gConfig.theme || 'default'}"
    )
    
    demo.launch(share=${gConfig.share ? 'True' : 'False'})

# Uncomment to run the app directly
# launch_gradio_app()
`;
  }

  if (streamlitNode) {
    const sConfig = streamlitNode.data.config;
    pythonCode += `
# ==========================================
# Streamlit Inference Interface
# ==========================================
# Save this code as app.py and run 'streamlit run app.py'

import streamlit as st
from transformers import pipeline

def launch_streamlit_app():
    st.set_page_config(page_title="${sConfig.title || 'NeuroFlow App'}", layout="${sConfig.layout || 'centered'}")
    
    st.title("${sConfig.title || 'NeuroFlow App'}")
    
    # Load Model
    model_id = "./${projectName}" if os.path.exists("./${projectName}") else "${baseModel}"
    
    @st.cache_resource
    def load_pipeline():
        return pipeline("text-generation", model=model_id, device_map="auto")

    generator = load_pipeline()

    ${sConfig.enable_sidebar ? `
    with st.sidebar:
        st.header("Configuration")
        max_length = st.slider("Max Length", 50, 1024, 200)
        temperature = st.slider("Temperature", 0.1, 1.0, 0.7)
    ` : 'max_length = 200\n    temperature = 0.7'}

    ${sConfig.enable_chat ? `
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("What is up?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            response = generator(prompt, max_new_tokens=max_length, temperature=temperature)[0]['generated_text']
            st.markdown(response)
        
        st.session_state.messages.append({"role": "assistant", "content": response})
    ` : `
    user_input = st.text_area("Enter prompt:")
    if st.button("Generate"):
        with st.spinner("Thinking..."):
            result = generator(user_input, max_new_tokens=max_length, temperature=temperature)
            st.write(result[0]['generated_text'])
    `}
`;
  }

  // 5. Generate CLI Command
  // Only relevant if training is involved
  let cliCommand = "";
  if (trainerNode && config.params) {
    cliCommand = `autotrain llm --train \\
    --project-name "${config.project_name}" \\
    --model "${config.base_model}" \\
    --data-path "${config.data?.path}" \\
    --text-column "${config.data?.column_mapping?.text_column}" \\
    --lr ${config.params.learning_rate} \\
    --batch-size ${config.params.batch_size} \\
    --epochs ${config.params.num_train_epochs} \\
    --block-size ${config.params.block_size} \\
    ${config.params.peft ? '--use-peft \\' : '\\'}
    --lora-r ${config.params.lora_r} \\
    --mixed-precision ${config.params.mixed_precision}`;
  }

  return { config, cliCommand, pythonCode, validationErrors: [] };
};