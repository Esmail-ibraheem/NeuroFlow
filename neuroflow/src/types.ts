export enum NodeType {
  // Training
  DATASET = 'dataset',
  MODEL = 'model',
  TASK = 'task',
  TRAINER = 'trainer',
  HARDWARE = 'hardware',
  DEPLOY = 'deploy',

  // Inference & Engine
  INFERENCE = 'inference', // Run LLM (OpenAI, Gemini, Local)
  PROMPT = 'prompt',       // Prompt Template
  ROUTER = 'router',       // Logic (If/Else)
  
  // RAG & Memory
  VECTOR_DB = 'vector_db', 
  PDF_LOADER = 'pdf_loader',
  TEXT_SPLITTER = 'text_splitter',
  EMBEDDINGS = 'embeddings',

  // Tools & Agents
  TOOL = 'tool', // Generic Tool
  GOOGLE_SEARCH = 'google_search',
  PYTHON_REPL = 'python_repl',
  JAVASCRIPT = 'javascript', // New: JS Execution
  WEBHOOK = 'webhook',       // New: Webhook Trigger
  EMAIL = 'email',
  API_CALL = 'api_call',

  // App Interfaces
  GRADIO_UI = 'gradio_ui',
  STREAMLIT_UI = 'streamlit_ui',
}

export enum TaskType {
  // LLM
  LLM_SFT = 'llm-sft',
  LLM_ORPO = 'llm-orpo',
  LLM_GENERIC = 'llm-generic',
  LLM_DPO = 'llm-dpo',
  LLM_REWARD = 'llm-reward',
  
  // VLM
  VLM_CAPTIONING = 'vlm-captioning',
  VLM_VQA = 'vlm-vqa',

  // Sentence Transformers
  ST_PAIR = 'st-pair',
  ST_PAIR_CLASS = 'st-pair-classification',
  ST_PAIR_SCORE = 'st-pair-scoring',
  ST_TRIPLET = 'st-triplet',
  ST_QA = 'st-question-answering',

  // Text
  TEXT_CLASS = 'text-classification',
  TEXT_REGRESSION = 'text-regression',
  EXTRACTIVE_QA = 'extractive-question-answering',
  SEQ2SEQ = 'sequence-to-sequence',
  TOKEN_CLASS = 'token-classification',

  // Image
  IMAGE_CLASS = 'image-classification',
  IMAGE_REGRESSION = 'image-regression',
  OBJECT_DETECTION = 'object-detection',

  // Tabular
  TABULAR_CLASS = 'tabular-classification',
  TABULAR_REGRESSION = 'tabular-regression',
  
  // DreamBooth
  DREAMBOOTH = 'dreambooth',
}

export interface NodeData {
  label: string;
  type: NodeType;
  config: Record<string, any>;
  status?: 'idle' | 'ready' | 'running' | 'completed' | 'error';
  [key: string]: any;
}

export interface GenerationResponse {
  nodes: any[];
  edges: any[];
  summary: string;
}

export const NODE_COLORS: Record<NodeType, string> = {
  [NodeType.DATASET]: '#10b981', // Emerald
  [NodeType.MODEL]: '#3b82f6',   // Blue
  [NodeType.TASK]: '#f59e0b',    // Amber
  [NodeType.TRAINER]: '#ef4444', // Red
  [NodeType.HARDWARE]: '#8b5cf6',// Violet
  [NodeType.DEPLOY]: '#ec4899',  // Pink
  
  // Engine Colors
  [NodeType.INFERENCE]: '#06b6d4', // Cyan
  [NodeType.PROMPT]: '#64748b',    // Slate
  [NodeType.ROUTER]: '#eab308',    // Yellow
  
  // RAG Colors
  [NodeType.VECTOR_DB]: '#d946ef', // Fuchsia
  [NodeType.PDF_LOADER]: '#ef4444', // Red-500 (Adobe color-ish)
  [NodeType.TEXT_SPLITTER]: '#84cc16', // Lime
  [NodeType.EMBEDDINGS]: '#8b5cf6', // Violet
  
  // Tool Colors
  [NodeType.TOOL]: '#f97316',      // Orange
  [NodeType.GOOGLE_SEARCH]: '#4285F4', // Google Blue
  [NodeType.PYTHON_REPL]: '#FACC15', // Python Yellow
  [NodeType.JAVASCRIPT]: '#F7DF1E', // JS Yellow
  [NodeType.WEBHOOK]: '#E11D48', // Rose
  [NodeType.EMAIL]: '#14b8a6', // Teal
  [NodeType.API_CALL]: '#6366f1', // Indigo

  // App UI Colors
  [NodeType.GRADIO_UI]: '#F97316', // Gradio Orange
  [NodeType.STREAMLIT_UI]: '#FF4B4B', // Streamlit Red
};

export const NODE_ICONS: Record<NodeType, string> = {
  [NodeType.DATASET]: 'Database',
  [NodeType.MODEL]: 'Box',
  [NodeType.TASK]: 'Target',
  [NodeType.TRAINER]: 'Zap',
  [NodeType.HARDWARE]: 'Cpu',
  [NodeType.DEPLOY]: 'Rocket',
  
  // Engine Icons
  [NodeType.INFERENCE]: 'BrainCircuit',
  [NodeType.PROMPT]: 'MessageSquareText',
  [NodeType.ROUTER]: 'Split',
  
  // RAG Icons
  [NodeType.VECTOR_DB]: 'Library',
  [NodeType.PDF_LOADER]: 'FileText',
  [NodeType.TEXT_SPLITTER]: 'Scissors',
  [NodeType.EMBEDDINGS]: 'Binary',
  
  // Tool Icons
  [NodeType.TOOL]: 'Wrench',
  [NodeType.GOOGLE_SEARCH]: 'Search',
  [NodeType.PYTHON_REPL]: 'Terminal',
  [NodeType.JAVASCRIPT]: 'Code',
  [NodeType.WEBHOOK]: 'Webhook',
  [NodeType.EMAIL]: 'Mail',
  [NodeType.API_CALL]: 'Globe',

  // App UI Icons
  [NodeType.GRADIO_UI]: 'AppWindow',
  [NodeType.STREAMLIT_UI]: 'LayoutDashboard',
};