import React from 'react';
import { NodeType, NODE_ICONS, NODE_COLORS } from '../types';
import * as LucideIcons from 'lucide-react';

const Sidebar: React.FC = () => {
  const onDragStart = (event: React.DragEvent, nodeType: NodeType) => {
    event.dataTransfer.setData('application/reactflow', nodeType);
    event.dataTransfer.effectAllowed = 'move';
  };

  const categories = [
    { title: "Training Ops", items: [NodeType.TASK, NodeType.MODEL, NodeType.TRAINER, NodeType.HARDWARE, NodeType.DEPLOY] },
    { title: "Input & Data", items: [NodeType.DATASET, NodeType.WEBHOOK, NodeType.PROMPT] },
    { title: "AI Engine", items: [NodeType.INFERENCE, NodeType.ROUTER] },
    { title: "RAG & Memory", items: [NodeType.PDF_LOADER, NodeType.TEXT_SPLITTER, NodeType.EMBEDDINGS, NodeType.VECTOR_DB] },
    { title: "Tools & Scripting", items: [NodeType.GOOGLE_SEARCH, NodeType.PYTHON_REPL, NodeType.JAVASCRIPT, NodeType.API_CALL, NodeType.EMAIL] },
    { title: "App Interfaces", items: [NodeType.GRADIO_UI, NodeType.STREAMLIT_UI] },
  ];

  return (
    <aside className="w-64 bg-[#121214] border-r border-[#27272a] flex flex-col h-full text-sm select-none z-20">
      <div className="p-4 border-b border-[#27272a]">
        <h2 className="font-bold text-gray-100 flex items-center gap-2">
          <LucideIcons.Layers size={18} />
          Node Library
        </h2>
      </div>
      
      <div className="flex-1 overflow-y-auto p-4 space-y-6 scrollbar-thin">
        {categories.map((cat) => (
          <div key={cat.title}>
            <h3 className="text-[11px] font-bold text-gray-500 uppercase tracking-wider mb-2 pl-1">{cat.title}</h3>
            <div className="space-y-2">
              {cat.items.map((type) => {
                // Dynamic Icon Lookup
                const iconName = NODE_ICONS[type];
                const Icon = (LucideIcons as any)[iconName] || LucideIcons.Box;
                const color = NODE_COLORS[type];

                return (
                  <div
                    key={type}
                    onDragStart={(event) => onDragStart(event, type)}
                    draggable
                    className="group flex items-center gap-3 p-2 rounded bg-[#1e1e20] border border-[#27272a] hover:bg-[#27272a] hover:border-gray-500 cursor-grab active:cursor-grabbing transition-all shadow-sm"
                  >
                    <div 
                      className="w-8 h-8 rounded flex items-center justify-center text-white shadow-inner"
                      style={{ backgroundColor: color }}
                    >
                      <Icon size={16} />
                    </div>
                    <div className="flex flex-col">
                      <span className="font-medium text-gray-200 capitalize">{type.replace(/_/g, ' ')}</span>
                      <span className="text-[10px] text-gray-500">Drag to add</span>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        ))}
      </div>
      
      <div className="p-4 border-t border-[#27272a] bg-[#09090b]">
        <div className="flex items-center gap-2 text-xs text-gray-500">
          <LucideIcons.Info size={14} />
          <span>Drag nodes to the canvas</span>
        </div>
      </div>
    </aside>
  );
};

export default Sidebar;