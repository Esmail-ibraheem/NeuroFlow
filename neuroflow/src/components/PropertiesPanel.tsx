import React, { useEffect, useState } from 'react';
import { Node } from 'reactflow';
import { NodeData, NodeType, TaskType } from '../types';
import { Settings2, Trash2, X, Database, Layers, FolderOpen, Box, BrainCircuit, Split, FileText, Scissors, Binary, Search, Terminal, Mail, Globe, Library, AppWindow, LayoutDashboard, Code, Webhook, Loader2 } from 'lucide-react';
import { uploadNeuroflowFile } from '../services/backendService';

interface PropertiesPanelProps {
  selectedNode: Node<NodeData> | null;
  onUpdateNode: (id: string, data: NodeData) => void;
  onDeleteNode: (id: string) => void;
  onClose: () => void;
}

const MODEL_OPTIONS = [
  // Llama 3
  "meta-llama/Meta-Llama-3-8B",
  "meta-llama/Meta-Llama-3-8B-Instruct",
  "meta-llama/Meta-Llama-3-70B",

  // Llama 2
  "meta-llama/Llama-2-7b-hf",
  "meta-llama/Llama-2-13b-hf",
  "meta-llama/Llama-2-70b-hf",

  // Mistral / Mixtral
  "mistralai/Mistral-7B-v0.1",
  "mistralai/Mistral-7B-v0.3",
  "mistralai/Mistral-7B-Instruct-v0.3",
  "mistralai/Mixtral-8x7B-v0.1",
  "mistralai/Mixtral-8x22B-v0.1",

  // Gemma
  "google/gemma-2b",
  "google/gemma-7b",
  "google/gemma-1.1-7b-it",

  // Phi
  "microsoft/phi-2",
  "microsoft/Phi-3-mini-4k-instruct",
  "microsoft/Phi-3-medium-4k-instruct",
  "microsoft/Phi-3-vision-128k-instruct",

  // Qwen
  "Qwen/Qwen2-0.5B",
  "Qwen/Qwen2-1.5B",
  "Qwen/Qwen2-7B",
  "Qwen/Qwen2-72B",
  "Qwen/Qwen1.5-110B",

  // Yi
  "01-ai/Yi-6B",
  "01-ai/Yi-34B",
  "01-ai/Yi-1.5-9B",

  // Falcon
  "tiiuae/falcon-7b",
  "tiiuae/falcon-40b",
  "tiiuae/falcon-180B",

  // BERT / RoBERTa (Encoders)
  "bert-base-uncased",
  "bert-large-uncased",
  "roberta-base",
  "roberta-large",
  "distilbert-base-uncased",
  "facebook/bart-large",
  "xlm-roberta-base",

  // T5 (Seq2Seq)
  "google/t5-v1_1-base",
  "google/flan-t5-xl",
  "google/flan-t5-xxl",

  // Vision
  "google/vit-base-patch16-224",
  "microsoft/resnet-50",
  "facebook/detr-resnet-50",
  "openai/clip-vit-base-patch32",

  // Audio
  "openai/whisper-large-v3",
  "openai/whisper-medium",
  "openai/whisper-tiny",

  // Stable Diffusion
  "runwayml/stable-diffusion-v1-5",
  "stabilityai/stable-diffusion-xl-base-1.0"
];

const PropertiesPanel: React.FC<PropertiesPanelProps> = ({ selectedNode, onUpdateNode, onDeleteNode, onClose }) => {
  const [formData, setFormData] = useState<Record<string, any>>({});
  const [label, setLabel] = useState("");
  const [activeTab, setActiveTab] = useState<'general' | 'lora' | 'advanced'>('general');
  const [isUploading, setIsUploading] = useState(false);

  useEffect(() => {
    if (selectedNode) {
      setFormData(selectedNode.data.config || {});
      setLabel(selectedNode.data.label);
      setActiveTab('general'); // Reset tab on node change
    }
  }, [selectedNode]);

  if (!selectedNode) {
    return null;
  }

  const handleConfigChange = (key: string, value: any) => {
    const updated = { ...formData, [key]: value };
    setFormData(updated);
    onUpdateNode(selectedNode.id, {
      ...selectedNode.data,
      config: updated,
    });
  };

  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    setIsUploading(true);
    try {
      const result = await uploadNeuroflowFile(file);
      if (result.success) {
        handleConfigChange('path', result.path);
        handleConfigChange('local_filename', result.filename);
      }
    } catch (err: any) {
      alert(`Upload failed: ${err.message}`);
    } finally {
      setIsUploading(false);
      // Reset input so the same file can be uploaded again if needed
      e.target.value = '';
    }
  };

  const handleLabelChange = (val: string) => {
    setLabel(val);
    onUpdateNode(selectedNode.id, { ...selectedNode.data, label: val });
  };

  // Render specific inputs based on node type
  const renderFields = () => {
    switch (selectedNode.data.type) {
      // --- APP INTERFACES ---
      case NodeType.GRADIO_UI:
        return (
          <div className="space-y-4">
            <div className="bg-[#18181b] p-3 rounded border border-[#27272a]">
              <h4 className="text-xs font-bold text-gray-400 uppercase mb-3 flex items-center gap-2"><AppWindow size={12} /> Gradio Config</h4>
              <InputField label="App Title" value={formData.title || 'My AI App'} onChange={(v) => handleConfigChange('title', v)} />
              <SelectField
                label="Theme"
                value={formData.theme || 'default'}
                onChange={(v) => handleConfigChange('theme', v)}
                options={['default', 'huggingface', 'glass', 'soft']}
              />
              <div className="grid grid-cols-2 gap-2">
                <SelectField
                  label="Input Type"
                  value={formData.input_type || 'text'}
                  onChange={(v) => handleConfigChange('input_type', v)}
                  options={['text', 'image', 'audio', 'file']}
                />
                <SelectField
                  label="Output Type"
                  value={formData.output_type || 'text'}
                  onChange={(v) => handleConfigChange('output_type', v)}
                  options={['text', 'label', 'image', 'json']}
                />
              </div>
              <div className="flex items-center gap-2 mt-2">
                <input type="checkbox" checked={formData.share !== false} onChange={(e) => handleConfigChange('share', e.target.checked)} className="rounded border-gray-700 bg-[#09090b]" />
                <span className="text-xs text-gray-400">Create Public Share Link</span>
              </div>
            </div>
          </div>
        );

      case NodeType.STREAMLIT_UI:
        return (
          <div className="space-y-4">
            <div className="bg-[#18181b] p-3 rounded border border-[#27272a]">
              <h4 className="text-xs font-bold text-gray-400 uppercase mb-3 flex items-center gap-2"><LayoutDashboard size={12} /> Streamlit Config</h4>
              <InputField label="Page Title" value={formData.title || 'NeuroFlow App'} onChange={(v) => handleConfigChange('title', v)} />
              <SelectField
                label="Layout"
                value={formData.layout || 'centered'}
                onChange={(v) => handleConfigChange('layout', v)}
                options={['centered', 'wide']}
              />
              <div className="space-y-2 mt-2">
                <div className="flex items-center justify-between">
                  <span className="text-xs text-gray-400">Chat Interface</span>
                  <input type="checkbox" checked={formData.enable_chat || true} onChange={(e) => handleConfigChange('enable_chat', e.target.checked)} className="rounded border-gray-700 bg-[#09090b]" />
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-xs text-gray-400">Sidebar</span>
                  <input type="checkbox" checked={formData.enable_sidebar || true} onChange={(e) => handleConfigChange('enable_sidebar', e.target.checked)} className="rounded border-gray-700 bg-[#09090b]" />
                </div>
              </div>
            </div>
          </div>
        );

      // --- RAG & MEMORY NODES ---
      case NodeType.PDF_LOADER:
        return (
          <div className="space-y-4">
            <div className="bg-[#18181b] p-3 rounded border border-[#27272a]">
              <h4 className="text-xs font-bold text-gray-400 uppercase mb-3 flex items-center gap-2"><FileText size={12} /> PDF Source</h4>
              <InputField label="File URL" value={formData.url || ''} onChange={(v) => handleConfigChange('url', v)} placeholder="https://example.com/doc.pdf" />
              <div className="flex items-center gap-2 mt-2">
                <input type="checkbox" checked={formData.extract_images || false} onChange={(e) => handleConfigChange('extract_images', e.target.checked)} className="rounded border-gray-700 bg-[#09090b]" />
                <span className="text-xs text-gray-400">OCR / Extract Images</span>
              </div>
            </div>
          </div>
        );

      case NodeType.TEXT_SPLITTER:
        return (
          <div className="space-y-4">
            <div className="bg-[#18181b] p-3 rounded border border-[#27272a]">
              <h4 className="text-xs font-bold text-gray-400 uppercase mb-3 flex items-center gap-2"><Scissors size={12} /> Split Config</h4>
              <InputField label="Chunk Size" type="number" value={formData.chunk_size || 1000} onChange={(v) => handleConfigChange('chunk_size', v)} />
              <InputField label="Chunk Overlap" type="number" value={formData.chunk_overlap || 200} onChange={(v) => handleConfigChange('chunk_overlap', v)} />
              <SelectField
                label="Separator"
                value={formData.separator || 'newline'}
                onChange={(v) => handleConfigChange('separator', v)}
                options={['newline', 'double_newline', 'space', 'markdown_header']}
              />
            </div>
          </div>
        );

      case NodeType.EMBEDDINGS:
        return (
          <div className="space-y-4">
            <div className="bg-[#18181b] p-3 rounded border border-[#27272a]">
              <h4 className="text-xs font-bold text-gray-400 uppercase mb-3 flex items-center gap-2"><Binary size={12} /> Model Config</h4>
              <SelectField
                label="Provider"
                value={formData.provider || 'openai'}
                onChange={(v) => handleConfigChange('provider', v)}
                options={['openai', 'huggingface', 'cohere', 'google-gecko']}
              />
              <InputField label="Model ID" value={formData.model_id || 'text-embedding-3-small'} onChange={(v) => handleConfigChange('model_id', v)} />
              <InputField label="Dimensions" type="number" value={formData.dimensions || 1536} onChange={(v) => handleConfigChange('dimensions', v)} />
            </div>
          </div>
        );

      case NodeType.VECTOR_DB:
        return (
          <div className="space-y-4">
            <div className="bg-[#18181b] p-3 rounded border border-[#27272a]">
              <h4 className="text-xs font-bold text-gray-400 uppercase mb-3 flex items-center gap-2"><Library size={12} /> Store Config</h4>
              <SelectField
                label="Database Provider"
                value={formData.provider || 'pinecone'}
                onChange={(v) => handleConfigChange('provider', v)}
                options={['pinecone', 'chroma', 'weaviate', 'pgvector']}
              />
              <InputField label="Index Name" value={formData.index_name || ''} onChange={(v) => handleConfigChange('index_name', v)} />
              <InputField label="Namespace" value={formData.namespace || ''} onChange={(v) => handleConfigChange('namespace', v)} />
              <InputField label="API Key" type="password" value={formData.api_key || ''} onChange={(v) => handleConfigChange('api_key', v)} placeholder="DB_KEY..." />
            </div>
          </div>
        );

      // --- TOOLS & AGENTS NODES ---
      case NodeType.WEBHOOK:
        return (
          <div className="space-y-4">
            <div className="bg-[#18181b] p-3 rounded border border-[#27272a]">
              <h4 className="text-xs font-bold text-gray-400 uppercase mb-3 flex items-center gap-2"><Webhook size={12} /> Trigger Config</h4>
              <SelectField
                label="HTTP Method"
                value={formData.method || 'POST'}
                onChange={(v) => handleConfigChange('method', v)}
                options={['GET', 'POST', 'PUT']}
              />
              <InputField label="Path / Route" value={formData.path || '/webhook/my-trigger'} onChange={(v) => handleConfigChange('path', v)} />
              <div className="mt-2 text-[10px] text-gray-400">
                Listens for external requests to trigger this workflow.
              </div>
            </div>
          </div>
        );

      case NodeType.JAVASCRIPT:
        return (
          <div className="space-y-4">
            <div className="bg-[#18181b] p-3 rounded border border-[#27272a]">
              <h4 className="text-xs font-bold text-gray-400 uppercase mb-3 flex items-center gap-2"><Code size={12} /> Script Execution</h4>
              <textarea
                value={formData.code || '// Access inputs via "input"\n// Return objects to pass data\nreturn { processed: input.text };'}
                onChange={(e) => handleConfigChange('code', e.target.value)}
                className="w-full h-40 bg-[#09090b] text-gray-200 border border-[#27272a] rounded px-2 py-1.5 text-sm focus:outline-none focus:border-blue-500 transition font-mono focus:ring-1 ring-blue-500/50 mb-2"
              />
              <div className="flex items-center gap-2">
                <input type="checkbox" checked={formData.run_once || false} onChange={(e) => handleConfigChange('run_once', e.target.checked)} className="rounded border-gray-700 bg-[#09090b]" />
                <span className="text-xs text-gray-400">Run only once (not per item)</span>
              </div>
            </div>
          </div>
        );

      case NodeType.GOOGLE_SEARCH:
        return (
          <div className="space-y-4">
            <div className="bg-[#18181b] p-3 rounded border border-[#27272a]">
              <h4 className="text-xs font-bold text-gray-400 uppercase mb-3 flex items-center gap-2"><Search size={12} /> Search Tool</h4>
              <InputField label="Query Template" value={formData.query || ''} onChange={(v) => handleConfigChange('query', v)} placeholder="{{input}} or 'Latest news on...'" />
              <InputField label="Max Results" type="number" value={formData.limit || 5} onChange={(v) => handleConfigChange('limit', v)} />
              <SelectField
                label="Type"
                value={formData.search_type || 'web'}
                onChange={(v) => handleConfigChange('search_type', v)}
                options={['web', 'news', 'images', 'video']}
              />
            </div>
          </div>
        );

      case NodeType.PYTHON_REPL:
        return (
          <div className="space-y-4">
            <div className="bg-[#18181b] p-3 rounded border border-[#27272a]">
              <h4 className="text-xs font-bold text-gray-400 uppercase mb-3 flex items-center gap-2"><Terminal size={12} /> Python Execution</h4>
              <textarea
                value={formData.code || ''}
                onChange={(e) => handleConfigChange('code', e.target.value)}
                placeholder="print('Hello World') # Use {{input}} for dynamic vars"
                className="w-full h-32 bg-[#09090b] text-gray-200 border border-[#27272a] rounded px-2 py-1.5 text-sm focus:outline-none focus:border-blue-500 transition font-mono focus:ring-1 ring-blue-500/50 mb-2"
              />
              <div className="flex items-center gap-2">
                <input type="checkbox" checked={formData.safe_mode !== false} onChange={(e) => handleConfigChange('safe_mode', e.target.checked)} className="rounded border-gray-700 bg-[#09090b]" />
                <span className="text-xs text-gray-400">Safe Mode (Restricted Imports)</span>
              </div>
            </div>
          </div>
        );

      case NodeType.API_CALL:
        return (
          <div className="space-y-4">
            <div className="bg-[#18181b] p-3 rounded border border-[#27272a]">
              <h4 className="text-xs font-bold text-gray-400 uppercase mb-3 flex items-center gap-2"><Globe size={12} /> API Request</h4>
              <SelectField
                label="Method"
                value={formData.method || 'GET'}
                onChange={(v) => handleConfigChange('method', v)}
                options={['GET', 'POST', 'PUT', 'DELETE', 'PATCH']}
              />
              <InputField label="URL" value={formData.url || ''} onChange={(v) => handleConfigChange('url', v)} placeholder="https://api.example.com/v1/..." />
              <div className="mt-2">
                <label className="block text-xs font-medium text-gray-400 mb-1">Headers (JSON)</label>
                <textarea
                  value={formData.headers || '{"Content-Type": "application/json"}'}
                  onChange={(e) => handleConfigChange('headers', e.target.value)}
                  className="w-full h-16 bg-[#09090b] text-gray-200 border border-[#27272a] rounded px-2 py-1.5 text-xs font-mono focus:outline-none focus:border-blue-500"
                />
              </div>
              <div className="mt-2">
                <label className="block text-xs font-medium text-gray-400 mb-1">Body (JSON)</label>
                <textarea
                  value={formData.body || '{}'}
                  onChange={(e) => handleConfigChange('body', e.target.value)}
                  className="w-full h-16 bg-[#09090b] text-gray-200 border border-[#27272a] rounded px-2 py-1.5 text-xs font-mono focus:outline-none focus:border-blue-500"
                />
              </div>
            </div>
          </div>
        );

      case NodeType.EMAIL:
        return (
          <div className="space-y-4">
            <div className="bg-[#18181b] p-3 rounded border border-[#27272a]">
              <h4 className="text-xs font-bold text-gray-400 uppercase mb-3 flex items-center gap-2"><Mail size={12} /> SMTP Config</h4>
              <InputField label="Recipient" value={formData.to || ''} onChange={(v) => handleConfigChange('to', v)} placeholder="email@example.com" />
              <InputField label="Subject" value={formData.subject || ''} onChange={(v) => handleConfigChange('subject', v)} placeholder="Summary: {{input}}" />
              <InputField label="SMTP Host" value={formData.smtp_host || 'smtp.gmail.com'} onChange={(v) => handleConfigChange('smtp_host', v)} />
              <InputField label="SMTP Port" type="number" value={formData.smtp_port || 587} onChange={(v) => handleConfigChange('smtp_port', v)} />
            </div>
          </div>
        );

      // --- EXISTING NODES (Retained) ---
      case NodeType.INFERENCE:
        return (
          <div className="space-y-4">
            <div className="bg-[#18181b] p-3 rounded border border-[#27272a]">
              <h4 className="text-xs font-bold text-gray-400 uppercase mb-3 flex items-center gap-2"><BrainCircuit size={12} /> Model Config</h4>
              <div className="space-y-3">
                <SelectField
                  label="Provider"
                  value={formData.provider || 'openai'}
                  onChange={(v) => handleConfigChange('provider', v)}
                  options={['openai', 'anthropic', 'gemini', 'local_vllm', 'huggingface_api']}
                />
                <InputField label="Model Name" value={formData.model || 'gpt-4-turbo'} onChange={(v) => handleConfigChange('model', v)} placeholder="e.g. gpt-4" />
                <InputField label="API Key" type="password" value={formData.api_key || ''} onChange={(v) => handleConfigChange('api_key', v)} placeholder="sk-..." />
                <InputField label="Temperature" type="number" value={formData.temperature || 0.7} onChange={(v) => handleConfigChange('temperature', v)} />
                <InputField label="Max Tokens" type="number" value={formData.max_tokens || 2048} onChange={(v) => handleConfigChange('max_tokens', v)} />
              </div>
            </div>

            <div>
              <label className="block text-xs font-medium text-gray-400 mb-1">System Prompt</label>
              <textarea
                value={formData.system_prompt || ''}
                onChange={(e) => handleConfigChange('system_prompt', e.target.value)}
                placeholder="You are a helpful assistant..."
                className="w-full h-24 bg-[#09090b] text-gray-200 border border-[#27272a] rounded px-2 py-1.5 text-sm focus:outline-none focus:border-blue-500 transition font-mono focus:ring-1 ring-blue-500/50"
              />
            </div>
          </div>
        );

      case NodeType.PROMPT:
        return (
          <div className="space-y-4">
            <div>
              <label className="block text-xs font-medium text-gray-400 mb-1">Prompt Template</label>
              <textarea
                value={formData.template || ''}
                onChange={(e) => handleConfigChange('template', e.target.value)}
                placeholder="Summarize the following text: {{input}}"
                className="w-full h-32 bg-[#09090b] text-gray-200 border border-[#27272a] rounded px-2 py-1.5 text-sm focus:outline-none focus:border-blue-500 transition font-mono focus:ring-1 ring-blue-500/50"
              />
              <div className="text-[10px] text-gray-500 mt-1">Use {'{{variable}}'} to insert inputs from previous nodes.</div>
            </div>
            <InputField label="Output Format" value={formData.output_format || 'text'} onChange={(v) => handleConfigChange('output_format', v)} placeholder="text or json" />
          </div>
        );

      case NodeType.ROUTER:
        return (
          <div className="space-y-4">
            <div className="bg-[#18181b] p-3 rounded border border-[#27272a]">
              <h4 className="text-xs font-bold text-gray-400 uppercase mb-3 flex items-center gap-2"><Split size={12} /> Routing Logic</h4>
              <div className="space-y-3">
                <SelectField
                  label="Condition Type"
                  value={formData.condition_type || 'contains'}
                  onChange={(v) => handleConfigChange('condition_type', v)}
                  options={['contains', 'equals', 'starts_with', 'regex', 'javascript']}
                />
                <InputField label="Value to Match" value={formData.match_value || ''} onChange={(v) => handleConfigChange('match_value', v)} />

                <div className="p-2 bg-[#27272a] rounded text-[10px] text-gray-400">
                  If the input matches, data flows to the <strong>True</strong> output. Otherwise, it flows to <strong>False</strong>.
                </div>
              </div>
            </div>
          </div>
        );

      case NodeType.DATASET:
        return (
          <div className="space-y-4">
            <div className="bg-[#18181b] p-3 rounded border border-[#27272a]">
              <h4 className="text-xs font-bold text-gray-400 uppercase mb-3 flex items-center gap-2"><Database size={12} /> Source</h4>
              <div className="space-y-3">
                <SelectField
                  label="Source Type"
                  value={formData.source_type || 'hub'}
                  onChange={(v) => handleConfigChange('source_type', v)}
                  options={['hub', 'local']}
                  displayOptions={{ 'hub': 'Hugging Face Hub', 'local': 'Local / Custom Upload' }}
                />

                {(!formData.source_type || formData.source_type === 'hub') && (
                  <>
                    <InputField label="Hugging Face Repo ID" value={formData.path || ''} onChange={(v) => handleConfigChange('path', v)} placeholder="user/dataset" />
                    <InputField label="Configuration / Subset" value={formData.config_name || 'default'} onChange={(v) => handleConfigChange('config_name', v)} />
                  </>
                )}

                {formData.source_type === 'local' && (
                  <div
                    className={`p-3 border border-dashed rounded bg-[#1e1e20] text-center cursor-pointer transition ${isUploading ? 'opacity-50 border-blue-500' : 'border-gray-600 hover:border-gray-400'}`}
                    onClick={() => !isUploading && document.getElementById('local-file-input')?.click()}
                  >
                    {isUploading ? (
                      <>
                        <Loader2 size={20} className="mx-auto text-blue-500 mb-2 animate-spin" />
                        <div className="text-xs text-blue-400">Uploading...</div>
                      </>
                    ) : (
                      <>
                        <FolderOpen size={20} className="mx-auto text-gray-400 mb-2" />
                        <div className="text-xs text-gray-300">
                          {formData.local_filename ? `File: ${formData.local_filename}` : 'Click to upload file'}
                        </div>
                        <div className="text-[10px] text-gray-500 mt-1">.csv, .jsonl, .parquet</div>
                      </>
                    )}
                    <input
                      id="local-file-input"
                      type="file"
                      className="hidden"
                      onChange={handleFileUpload}
                      accept=".csv,.jsonl,.json,.parquet"
                    />
                  </div>
                )}

                <div className="grid grid-cols-2 gap-2">
                  <InputField label="Train Split" value={formData.split || 'train'} onChange={(v) => handleConfigChange('split', v)} />
                  <InputField label="Validation Split" value={formData.valid_split || ''} onChange={(v) => handleConfigChange('valid_split', v)} placeholder="validation / dev" />
                </div>
              </div>
            </div>

            <div className="bg-[#18181b] p-3 rounded border border-[#27272a]">
              <h4 className="text-xs font-bold text-gray-400 uppercase mb-3 flex items-center gap-2"><Layers size={12} /> Column Mapping</h4>
              <div className="space-y-3">
                <InputField label="Text Column" value={formData.text_column || 'text'} onChange={(v) => handleConfigChange('text_column', v)} placeholder="e.g. text, prompt" />
                <InputField label="Target Column" value={formData.target_column || ''} onChange={(v) => handleConfigChange('target_column', v)} placeholder="e.g. label, completion" />
                <InputField label="Prompt Column" value={formData.prompt_column || ''} onChange={(v) => handleConfigChange('prompt_column', v)} placeholder="prompt (for instruction datasets)" />
                <InputField label="Rejected Column" value={formData.rejected_text_column || ''} onChange={(v) => handleConfigChange('rejected_text_column', v)} placeholder="rejected (for DPO/ORPO)" />
                <InputField label="Question Column" value={formData.question_column || ''} onChange={(v) => handleConfigChange('question_column', v)} placeholder="question (QA tasks)" />
                <InputField label="Answer Column" value={formData.answer_column || ''} onChange={(v) => handleConfigChange('answer_column', v)} placeholder="answers (QA tasks)" />
                <InputField label="Sentence 1 Column" value={formData.sentence1_column || ''} onChange={(v) => handleConfigChange('sentence1_column', v)} placeholder="sentence1 (ST tasks)" />
                <InputField label="Sentence 2 Column" value={formData.sentence2_column || ''} onChange={(v) => handleConfigChange('sentence2_column', v)} placeholder="sentence2 (ST tasks)" />
                <InputField label="Sentence 3 Column" value={formData.sentence3_column || ''} onChange={(v) => handleConfigChange('sentence3_column', v)} placeholder="sentence3 (triplet)" />
                <div className="grid grid-cols-2 gap-2">
                  <InputField label="Tokens Column" value={formData.tokens_column || ''} onChange={(v) => handleConfigChange('tokens_column', v)} placeholder="tokens (NER)" />
                  <InputField label="Tags Column" value={formData.tags_column || ''} onChange={(v) => handleConfigChange('tags_column', v)} placeholder="tags (NER)" />
                </div>
                <div className="grid grid-cols-2 gap-2">
                  <InputField label="Image Column" value={formData.image_column || ''} onChange={(v) => handleConfigChange('image_column', v)} placeholder="image path" />
                  <InputField label="Objects/Boxes Column" value={formData.objects_column || ''} onChange={(v) => handleConfigChange('objects_column', v)} placeholder="objects (detection)" />
                </div>
                <div className="grid grid-cols-2 gap-2">
                  <InputField label="ID Column" value={formData.id_column || ''} onChange={(v) => handleConfigChange('id_column', v)} placeholder="id (tabular)" />
                  <InputField label="Target Columns (comma separated)" value={formData.target_columns || ''} onChange={(v) => handleConfigChange('target_columns', v)} placeholder="label_a,label_b" />
                </div>
                <div className="flex items-center gap-2 mt-2">
                  <input type="checkbox" checked={formData.train_test_split || false} onChange={(e) => handleConfigChange('train_test_split', e.target.checked)} className="rounded border-gray-700 bg-[#09090b]" />
                  <span className="text-xs text-gray-400">Auto Train/Test Split</span>
                </div>
              </div>
            </div>
          </div>
        );

      case NodeType.MODEL:
        return (
          <div className="space-y-4">
            <div className="bg-[#18181b] p-3 rounded border border-[#27272a]">
              <h4 className="text-xs font-bold text-gray-400 uppercase mb-3 flex items-center gap-2"><Box size={12} /> Model Selection</h4>
              <div>
                <label className="block text-xs font-medium text-gray-400 mb-1">Model ID</label>
                <div className="relative group">
                  <input
                    list="model-suggestions"
                    type="text"
                    value={formData.modelId || ''}
                    onChange={(e) => handleConfigChange('modelId', e.target.value)}
                    placeholder="Select or type custom ID..."
                    className="w-full bg-[#09090b] text-gray-200 border border-[#27272a] rounded px-2 py-2 text-sm focus:outline-none focus:border-blue-500 transition font-mono focus:ring-1 ring-blue-500/50"
                  />
                  <datalist id="model-suggestions">
                    {MODEL_OPTIONS.map((m) => (
                      <option key={m} value={m} />
                    ))}
                  </datalist>
                  <div className="text-[10px] text-gray-500 mt-2 flex items-center gap-1">
                    <span className="bg-[#27272a] px-1 rounded text-gray-400 border border-[#3f3f46]">Tip</span>
                    Type any Hugging Face model ID (e.g. 'user/my-model') to use a custom model.
                  </div>
                </div>
              </div>
            </div>

            <div className="space-y-3">
              <InputField label="Revision" value={formData.revision || 'main'} onChange={(v) => handleConfigChange('revision', v)} placeholder="Branch name or commit hash" />
              <SelectField
                label="Quantization"
                value={formData.quantization ?? 'none'}
                onChange={(v) => handleConfigChange('quantization', v === 'none' ? null : v)}
                options={['none', 'int4', 'int8']}
              />
              <SelectField
                label="Chat Template"
                value={formData.chat_template || 'none'}
                onChange={(v) => handleConfigChange('chat_template', v)}
                options={['none', 'chatml', 'zephyr', 'tokenizer', 'llama3']}
              />
            </div>
          </div>
        );

      case NodeType.TASK:
        return (
          <div className="space-y-4">
            <div>
              <label className="block text-xs font-medium text-gray-400 mb-1">Task Type</label>
              <select
                value={formData.task || TaskType.LLM_SFT}
                onChange={(e) => handleConfigChange('task', e.target.value)}
                className="w-full bg-[#09090b] text-gray-200 border border-[#27272a] rounded px-2 py-1.5 text-sm focus:outline-none focus:border-blue-500 transition appearance-none cursor-pointer"
              >
                <optgroup label="LLM Finetuning">
                  <option value={TaskType.LLM_SFT}>LLM SFT</option>
                  <option value={TaskType.LLM_ORPO}>LLM ORPO</option>
                  <option value={TaskType.LLM_GENERIC}>LLM Generic</option>
                  <option value={TaskType.LLM_DPO}>LLM DPO</option>
                  <option value={TaskType.LLM_REWARD}>LLM Reward</option>
                </optgroup>
                <optgroup label="VLM Finetuning">
                  <option value={TaskType.VLM_CAPTIONING}>VLM Captioning</option>
                  <option value={TaskType.VLM_VQA}>VLM VQA</option>
                </optgroup>
                <optgroup label="Sentence Transformers">
                  <option value={TaskType.ST_PAIR}>ST Pair</option>
                  <option value={TaskType.ST_PAIR_CLASS}>ST Pair Classification</option>
                  <option value={TaskType.ST_PAIR_SCORE}>ST Pair Scoring</option>
                  <option value={TaskType.ST_TRIPLET}>ST Triplet</option>
                  <option value={TaskType.ST_QA}>ST Question Answering</option>
                </optgroup>
                <optgroup label="Other Text Tasks">
                  <option value={TaskType.TEXT_CLASS}>Text Classification</option>
                  <option value={TaskType.TEXT_REGRESSION}>Text Regression</option>
                  <option value={TaskType.EXTRACTIVE_QA}>Extractive Question Answering</option>
                  <option value={TaskType.SEQ2SEQ}>Sequence to Sequence</option>
                  <option value={TaskType.TOKEN_CLASS}>Token Classification</option>
                </optgroup>
                <optgroup label="Image Tasks">
                  <option value={TaskType.IMAGE_CLASS}>Image Classification</option>
                  <option value={TaskType.IMAGE_REGRESSION}>Image Regression</option>
                  <option value={TaskType.OBJECT_DETECTION}>Object Detection</option>
                </optgroup>
                <optgroup label="Tabular Tasks">
                  <option value={TaskType.TABULAR_CLASS}>Tabular Classification</option>
                  <option value={TaskType.TABULAR_REGRESSION}>Tabular Regression</option>
                </optgroup>
                <optgroup label="Generation">
                  <option value={TaskType.DREAMBOOTH}>DreamBooth</option>
                </optgroup>
              </select>
            </div>
            <div className="text-xs text-gray-500 bg-[#18181b] p-2 rounded">
              Selected task determines the training pipeline structure.
            </div>
          </div>
        );

      case NodeType.TRAINER:
        return (
          <div className="flex flex-col h-full">
            {/* Tabs */}
            <div className="flex bg-[#18181b] rounded-md p-1 mb-4 gap-1">
              {(['general', 'lora', 'advanced'] as const).map(tab => (
                <button
                  key={tab}
                  onClick={() => setActiveTab(tab)}
                  className={`flex-1 py-1.5 text-xs font-medium rounded capitalize transition-colors ${activeTab === tab
                      ? 'bg-[#27272a] text-white shadow-sm'
                      : 'text-gray-500 hover:text-gray-300'
                    }`}
                >
                  {tab}
                </button>
              ))}
            </div>

            {/* Tab Content */}
            <div className="space-y-4 animate-in fade-in duration-300">
              {activeTab === 'general' && (
                <>
                  <InputField label="Learning Rate" type="number" value={formData.learning_rate || 2e-4} onChange={(v) => handleConfigChange('learning_rate', v)} />
                  <InputField label="Epochs" type="number" value={formData.num_train_epochs || 1} onChange={(v) => handleConfigChange('num_train_epochs', v)} />
                  <InputField label="Batch Size" type="number" value={formData.batch_size || 4} onChange={(v) => handleConfigChange('batch_size', v)} />
                  <InputField label="Block Size (Context)" type="number" value={formData.block_size || 1024} onChange={(v) => handleConfigChange('block_size', v)} />
                  <SelectField
                    label="Optimizer"
                    value={formData.optimizer || 'adamw_torch'}
                    onChange={(v) => handleConfigChange('optimizer', v)}
                    options={['adamw_torch', 'adamw_bnb_8bit', 'sgd', 'paged_adamw_8bit']}
                  />
                  <SelectField
                    label="Scheduler"
                    value={formData.scheduler || 'linear'}
                    onChange={(v) => handleConfigChange('scheduler', v)}
                    options={['linear', 'cosine', 'constant', 'polynomial']}
                  />
                </>
              )}

              {activeTab === 'lora' && (
                <div className="space-y-3">
                  <div className="flex items-center justify-between mb-2">
                    <label className="text-xs font-medium text-gray-400">Enable PEFT/LoRA</label>
                    <input type="checkbox" checked={formData.use_peft !== false} onChange={(e) => handleConfigChange('use_peft', e.target.checked)} className="rounded border-gray-700 bg-[#09090b] accent-blue-600" />
                  </div>
                  <div className={`space-y-3 ${formData.use_peft === false ? 'opacity-50 pointer-events-none' : ''}`}>
                    <InputField label="LoRA R" type="number" value={formData.lora_r || 16} onChange={(v) => handleConfigChange('lora_r', v)} />
                    <InputField label="LoRA Alpha" type="number" value={formData.lora_alpha || 32} onChange={(v) => handleConfigChange('lora_alpha', v)} />
                    <InputField label="LoRA Dropout" type="number" value={formData.lora_dropout || 0.05} onChange={(v) => handleConfigChange('lora_dropout', v)} />
                    <InputField label="Target Modules" value={formData.target_modules || 'all-linear'} onChange={(v) => handleConfigChange('target_modules', v)} placeholder="all-linear, q_proj,v_proj" />
                  </div>
                </div>
              )}

              {activeTab === 'advanced' && (
                <>
                  <SelectField
                    label="Mixed Precision"
                    value={formData.mixed_precision || 'fp16'}
                    onChange={(v) => handleConfigChange('mixed_precision', v)}
                    options={['no', 'fp16', 'bf16']}
                  />
                  <InputField label="Gradient Accumulation" type="number" value={formData.gradient_accumulation || 1} onChange={(v) => handleConfigChange('gradient_accumulation', v)} />
                  <InputField label="Warmup Ratio" type="number" value={formData.warmup_ratio || 0.1} onChange={(v) => handleConfigChange('warmup_ratio', v)} />
                  <InputField label="Weight Decay" type="number" value={formData.weight_decay || 0.01} onChange={(v) => handleConfigChange('weight_decay', v)} />
                  <div className="flex items-center justify-between pt-2">
                    <label className="text-xs font-medium text-gray-400">Gradient Checkpointing</label>
                    <input type="checkbox" checked={formData.gradient_checkpointing || true} onChange={(e) => handleConfigChange('gradient_checkpointing', e.target.checked)} className="rounded border-gray-700 bg-[#09090b] accent-blue-600" />
                  </div>
                </>
              )}
            </div>
          </div>
        );

      case NodeType.HARDWARE:
        return (
          <div className="space-y-4">
            <SelectField
              label="Compute Backend"
              value={formData.backend || 'spaces'}
              onChange={(v) => handleConfigChange('backend', v)}
              options={['Local Process', 'Hugging Face Spaces', 'DGX Cloud', 'AWS SageMaker']}
            />
            <SelectField
              label="Accelerator Type"
              value={formData.accelerator || 'A10G'}
              onChange={(v) => handleConfigChange('accelerator', v)}
              options={['T4 (Small)', 'A10G (Medium)', 'A100 (Large)', 'H100 (Massive)']}
            />
            <InputField label="Num GPUs" type="number" value={formData.num_gpus || 1} onChange={(v) => handleConfigChange('num_gpus', v)} />
          </div>
        );

      case NodeType.DEPLOY:
        return (
          <div className="space-y-4">
            <InputField label="Push to Hub Repo" value={formData.hubId || ''} onChange={(v) => handleConfigChange('hubId', v)} placeholder="username/model-name" />
            <SelectField
              label="Visibility"
              value={formData.private ? 'private' : 'public'}
              onChange={(v) => handleConfigChange('private', v === 'private')}
              options={['public', 'private']}
            />
            <InputField label="HF Token" type="password" value={formData.token || ''} onChange={(v) => handleConfigChange('token', v)} placeholder="hf_..." />
          </div>
        );
      default:
        return <div className="text-gray-500 text-sm italic">No specific properties for this node type.</div>;
    }
  };

  return (
    <div className="w-80 bg-[#121214] border-l border-[#27272a] h-full flex flex-col absolute right-0 top-0 z-20 shadow-2xl">
      <div className="p-4 border-b border-[#27272a] flex items-center justify-between bg-[#18181b]">
        <h2 className="font-bold text-gray-100 flex items-center gap-2">
          <Settings2 size={16} />
          Properties
        </h2>
        <button onClick={onClose} className="text-gray-500 hover:text-white transition">
          <X size={18} />
        </button>
      </div>

      <div className="flex-1 overflow-y-auto p-4 scrollbar-thin">
        <div className="space-y-4">
          <div>
            <label className="block text-xs font-semibold text-gray-500 uppercase mb-1">Node Label</label>
            <input
              type="text"
              value={label}
              onChange={(e) => handleLabelChange(e.target.value)}
              className="w-full bg-[#27272a] text-white border border-[#3f3f46] rounded px-3 py-2 text-sm focus:outline-none focus:border-blue-500 transition"
            />
          </div>

          <div className="h-px bg-[#27272a] w-full my-2" />

          <div className="h-full">
            {renderFields()}
          </div>
        </div>
      </div>

      <div className="p-4 border-t border-[#27272a] bg-[#09090b] flex gap-2">
        <button
          onClick={() => onDeleteNode(selectedNode.id)}
          className="flex-1 flex items-center justify-center gap-2 bg-red-500/10 text-red-500 hover:bg-red-500 hover:text-white px-4 py-2 rounded text-sm transition font-medium border border-red-500/20"
        >
          <Trash2 size={14} />
          Delete
        </button>
      </div>
    </div>
  );
};

// Helper Components
const InputField = ({ label, value, onChange, type = "text", placeholder = "" }: any) => (
  <div>
    <label className="block text-xs font-medium text-gray-400 mb-1">{label}</label>
    <input
      type={type}
      value={value}
      onChange={(e) => onChange(e.target.value)}
      placeholder={placeholder}
      className="w-full bg-[#09090b] text-gray-200 border border-[#27272a] rounded px-2 py-1.5 text-sm focus:outline-none focus:border-blue-500 transition font-mono focus:ring-1 ring-blue-500/50"
    />
  </div>
);

const SelectField = ({ label, value, onChange, options, displayOptions }: any) => (
  <div>
    <label className="block text-xs font-medium text-gray-400 mb-1">{label}</label>
    <select
      value={value}
      onChange={(e) => onChange(e.target.value)}
      className="w-full bg-[#09090b] text-gray-200 border border-[#27272a] rounded px-2 py-1.5 text-sm focus:outline-none focus:border-blue-500 transition appearance-none cursor-pointer"
    >
      {options.map((opt: string) => (
        <option key={opt} value={opt}>
          {displayOptions ? displayOptions[opt] : opt}
        </option>
      ))}
    </select>
  </div>
);

export default PropertiesPanel;
