import React, { memo } from 'react';
import { Handle, Position, NodeProps } from 'reactflow';
import { 
  Database, Box, Target, Zap, Cpu, Rocket, 
  BrainCircuit, MessageSquareText, Split, Library, Wrench,
  FileText, Scissors, Binary, Search, Terminal, Mail, Globe,
  AppWindow, LayoutDashboard, Code, Webhook
} from 'lucide-react';
import { NodeData, NODE_COLORS, NodeType, TaskType } from '../types';

const iconMap: Record<string, React.FC<any>> = {
  [NodeType.DATASET]: Database,
  [NodeType.MODEL]: Box,
  [NodeType.TASK]: Target,
  [NodeType.TRAINER]: Zap,
  [NodeType.HARDWARE]: Cpu,
  [NodeType.DEPLOY]: Rocket,
  
  // Engine Icons
  [NodeType.INFERENCE]: BrainCircuit,
  [NodeType.PROMPT]: MessageSquareText,
  [NodeType.ROUTER]: Split,
  
  // RAG Icons
  [NodeType.VECTOR_DB]: Library,
  [NodeType.PDF_LOADER]: FileText,
  [NodeType.TEXT_SPLITTER]: Scissors,
  [NodeType.EMBEDDINGS]: Binary,
  
  // Tool Icons
  [NodeType.TOOL]: Wrench,
  [NodeType.GOOGLE_SEARCH]: Search,
  [NodeType.PYTHON_REPL]: Terminal,
  [NodeType.JAVASCRIPT]: Code,
  [NodeType.WEBHOOK]: Webhook,
  [NodeType.EMAIL]: Mail,
  [NodeType.API_CALL]: Globe,

  // App UI Icons
  [NodeType.GRADIO_UI]: AppWindow,
  [NodeType.STREAMLIT_UI]: LayoutDashboard,
};

const CustomNode = ({ data, selected }: NodeProps<NodeData>) => {
  const Icon = iconMap[data.type] || Box;
  const color = NODE_COLORS[data.type] || '#71717a';

  // Helper to get important badges based on node type
  const getBadges = () => {
    const badges = [];
    if (data.type === NodeType.MODEL) {
      if (data.config.quantization && data.config.quantization !== 'none') badges.push(data.config.quantization);
    }
    if (data.type === NodeType.TRAINER) {
       if (data.config.use_peft !== false) badges.push('LoRA');
       if (data.config.mixed_precision && data.config.mixed_precision !== 'no') badges.push(data.config.mixed_precision);
    }
    if (data.type === NodeType.DATASET) {
        if (data.config.split) badges.push(data.config.split);
    }
    // Engine Badges
    if (data.type === NodeType.INFERENCE) {
        if (data.config.provider) badges.push(data.config.provider);
        if (data.config.temperature) badges.push(`T:${data.config.temperature}`);
    }
    if (data.type === NodeType.ROUTER) {
        badges.push('Logic');
    }
    // RAG Badges
    if (data.type === NodeType.TEXT_SPLITTER) {
        if (data.config.chunk_size) badges.push(`${data.config.chunk_size} chars`);
    }
    // Tool Badges
    if (data.type === NodeType.WEBHOOK) {
        if (data.config.method) badges.push(data.config.method);
    }
    if (data.type === NodeType.JAVASCRIPT) {
        badges.push('JS');
    }
    // UI Badges
    if (data.type === NodeType.GRADIO_UI || data.type === NodeType.STREAMLIT_UI) {
        badges.push('Web App');
    }
    return badges;
  };

  const badges = getBadges();

  return (
    <div 
      className={`min-w-[220px] rounded-lg shadow-xl bg-[#18181b] border-2 transition-all duration-200 overflow-hidden group
      ${selected ? 'border-white ring-2 ring-white/20' : 'border-[#27272a] hover:border-[#3f3f46]'}`}
    >
      {/* Header / Title Bar */}
      <div 
        className="px-3 py-2 flex items-center gap-2 text-white font-bold text-sm relative overflow-hidden"
        style={{
          background: `linear-gradient(90deg, ${color}dd 0%, ${color}88 100%)`
        }}
      >
        <Icon size={16} className="text-white drop-shadow-md" />
        <span className="drop-shadow-md tracking-wide uppercase text-[10px] opacity-90">{data.type.replace(/_/g, ' ')}</span>
        
        {/* Glossy shine effect */}
        <div className="absolute top-0 left-0 w-full h-[50%] bg-white/10 pointer-events-none" />
      </div>

      {/* Body */}
      <div className="p-3 relative bg-[#18181b]">
        <div className="text-sm font-semibold text-gray-100 mb-2 truncate" title={data.label}>{data.label}</div>
        
        {/* Badges Row */}
        {badges.length > 0 && (
             <div className="flex flex-wrap gap-1 mb-2">
                {badges.map(b => (
                    <span key={b} className="text-[9px] uppercase font-bold px-1.5 py-0.5 rounded bg-[#27272a] text-blue-400 border border-blue-900/30">
                        {b}
                    </span>
                ))}
             </div>
        )}

        {/* Key-Value Config Summary */}
        <div className="text-[10px] text-gray-400 space-y-1 font-mono">
           {Object.entries(data.config || {})
             .filter(([k]) => ['path', 'modelId', 'task', 'learning_rate', 'provider', 'model', 'condition', 'url', 'method', 'chunk_size', 'theme', 'layout'].includes(k))
             .slice(0, 4)
             .map(([key, val]) => (
             <div key={key} className="flex justify-between items-center border-b border-[#27272a] pb-1 last:border-0 last:pb-0">
               <span className="opacity-60 truncate max-w-[80px] capitalize">{key.replace(/_/g, ' ')}</span>
               <span className="text-gray-200 truncate max-w-[100px]">{String(val)}</span>
             </div>
           ))}
        </div>
      </div>

      {/* Handles */}
      <Handle
        type="target"
        position={Position.Left}
        className="!w-3 !h-3 !bg-[#3f3f46] !border-2 !border-[#18181b] rounded-full hover:!bg-white hover:!border-blue-500 transition-colors"
        style={{ left: -6 }}
      />
      <Handle
        type="source"
        position={Position.Right}
        className="!w-3 !h-3 !bg-[#3f3f46] !border-2 !border-[#18181b] rounded-full hover:!bg-white hover:!border-emerald-500 transition-colors"
        style={{ right: -6 }}
      />
    </div>
  );
};

export default memo(CustomNode);