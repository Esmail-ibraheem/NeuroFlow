import React, { useState, useCallback, useRef, useMemo } from 'react';
import ReactFlow, {
  Node,
  Edge,
  addEdge,
  Background,
  Controls,
  Connection,
  useNodesState,
  useEdgesState,
  ReactFlowProvider,
  BackgroundVariant,
  MiniMap
} from 'reactflow';
import Sidebar from './components/Sidebar';
import PropertiesPanel from './components/PropertiesPanel';
import CustomNode from './components/CustomNode';
import AIAssistant from './components/AIAssistant';
import TrainingMonitor from './components/TrainingMonitor';
import { compilePipeline } from './services/pipelineCompiler';
import { compileNeuroflowPipeline, runNeuroflowPipeline } from './services/backendService';
import { NodeData, NodeType, TaskType } from './types';
import { Play, Share2, Save, Terminal, Activity, AlertCircle } from 'lucide-react';

// Initial dummy state
const initialNodes: Node<NodeData>[] = [
  {
    id: '1',
    type: 'custom',
    position: { x: 100, y: 200 },
    data: { label: 'HF Dataset', type: NodeType.DATASET, config: { path: 'glue', split: 'train' } }
  },
  {
    id: '2',
    type: 'custom',
    position: { x: 450, y: 200 },
    data: { label: 'Llama 3 Base', type: NodeType.MODEL, config: { modelId: 'meta-llama/Meta-Llama-3-8B' } }
  },
  {
    id: '3',
    type: 'custom',
    position: { x: 450, y: 100 },
    data: { label: 'LLM SFT', type: NodeType.TASK, config: { task: 'llm-sft' } }
  },
  {
    id: '4',
    type: 'custom',
    position: { x: 800, y: 200 },
    data: { label: 'Trainer (LoRA)', type: NodeType.TRAINER, config: { learning_rate: 2e-4, use_peft: true } }
  }
];
const initialEdges: Edge[] = [];

const DEFAULT_NODE_CONFIG: Partial<Record<NodeType, Record<string, any>>> = {
  [NodeType.DATASET]: {
    source_type: 'hub',
    path: '',
    split: 'train',
    text_column: 'text',
    target_column: 'label',
    prompt_column: '',
    rejected_text_column: '',
  },
  [NodeType.MODEL]: {
    modelId: 'meta-llama/Meta-Llama-3-8B',
    quantization: 'none',
  },
  [NodeType.TASK]: {
    task: TaskType.LLM_SFT,
  },
  [NodeType.TRAINER]: {
    learning_rate: 2e-4,
    num_train_epochs: 1,
    batch_size: 4,
    block_size: 1024,
    optimizer: 'adamw_torch',
    scheduler: 'linear',
    mixed_precision: 'fp16',
    use_peft: true,
    lora_r: 16,
    lora_alpha: 32,
    lora_dropout: 0.05,
    target_modules: 'all-linear',
  },
};

const Flow = () => {
  const reactFlowWrapper = useRef<HTMLDivElement>(null);
  const [nodes, setNodes, onNodesChange] = useNodesState(initialNodes);
  const [edges, setEdges, onEdgesChange] = useEdgesState(initialEdges);
  const [selectedNode, setSelectedNode] = useState<Node<NodeData> | null>(null);
  const [reactFlowInstance, setReactFlowInstance] = useState<any>(null);

  // Execution State
  const [isSimulating, setIsSimulating] = useState(false);
  const [showMonitor, setShowMonitor] = useState(false);
  const [compiledData, setCompiledData] = useState<any>(null);
  const [validationError, setValidationError] = useState<string | null>(null);

  // Persistence: Load from localStorage on mount
  React.useEffect(() => {
    const saved = localStorage.getItem('neuroflow-graph');
    if (saved) {
      try {
        const { nodes: savedNodes, edges: savedEdges } = JSON.parse(saved);
        if (savedNodes && savedEdges) {
          setNodes(savedNodes);
          setEdges(savedEdges);
        }
      } catch (e) {
        console.error("Failed to load saved graph", e);
      }
    }
  }, [setNodes, setEdges]);

  const nodeTypes = useMemo(() => ({ custom: CustomNode }), []);

  const onConnect = useCallback((params: Connection) => setEdges((eds) => addEdge({
    ...params,
    animated: true,
    style: { stroke: '#52525b', strokeWidth: 2 }
  }, eds)), [setEdges]);

  const onDragOver = useCallback((event: React.DragEvent) => {
    event.preventDefault();
    event.dataTransfer.dropEffect = 'move';
  }, []);

  const onDrop = useCallback(
    (event: React.DragEvent) => {
      event.preventDefault();

      const type = event.dataTransfer.getData('application/reactflow') as NodeType;
      if (!type || !reactFlowWrapper.current || !reactFlowInstance) return;

      const position = reactFlowInstance.project({
        x: event.clientX - reactFlowWrapper.current.getBoundingClientRect().left,
        y: event.clientY - reactFlowWrapper.current.getBoundingClientRect().top,
      });

      const baseConfig = DEFAULT_NODE_CONFIG[type] ? { ...DEFAULT_NODE_CONFIG[type]! } : {};
      const newNode: Node<NodeData> = {
        id: `${type}-${Date.now()}`,
        type: 'custom',
        position,
        data: {
          label: `New ${type}`,
          type,
          config: baseConfig,
        },
      };

      setNodes((nds) => nds.concat(newNode));
    },
    [reactFlowInstance, setNodes]
  );

  const onNodeClick = (_: React.MouseEvent, node: Node) => {
    setSelectedNode(node);
  };

  const onPaneClick = () => {
    setSelectedNode(null);
  };

  const updateNodeData = (id: string, newData: NodeData) => {
    setNodes((nds) =>
      nds.map((node) => {
        if (node.id === id) {
          return { ...node, data: newData };
        }
        return node;
      })
    );
    if (selectedNode && selectedNode.id === id) {
      setSelectedNode({ ...selectedNode, data: newData });
    }
  };

  const deleteNode = (id: string) => {
    setNodes((nds) => nds.filter((n) => n.id !== id));
    setSelectedNode(null);
  };

  const handleRunSimulation = async () => {
    setValidationError(null);
    setIsSimulating(true);

    const deployNode = nodes.find((n) => n.data.type === NodeType.DEPLOY);
    const deployTokenRaw = deployNode?.data.config?.token?.trim();
    const deployToken = deployTokenRaw && deployTokenRaw.length > 0 ? deployTokenRaw : undefined;
    const username = deployNode?.data.config?.hubId?.split('/')?.[0];
    const hardwareNode = nodes.find((n) => n.data.type === NodeType.HARDWARE);
    const backendLabel =
      (hardwareNode?.data.config?.backend as string | undefined) || 'Local Process';
    const isLocalBackend = !hardwareNode || backendLabel.toLowerCase().startsWith('local');

    const fallbackToLocal = () => {
      const localResult = compilePipeline(nodes, edges);
      if (localResult.validationErrors.length > 0) {
        throw new Error(localResult.validationErrors[0]);
      }
      const projectName = localResult.config?.project_name;
      return { ...localResult, source: 'local', projectName };
    };

    try {
      let compiled;

      const shouldUseBackend = Boolean(deployToken) || isLocalBackend;

      if (!deployToken && !isLocalBackend) {
        throw new Error('Remote backends require a Deploy node with a Hugging Face token.');
      }

      if (shouldUseBackend) {
        try {
          compiled = await runNeuroflowPipeline(nodes, edges, deployToken, username);
        } catch (backendErr: any) {
          setValidationError(backendErr?.message || 'Backend run failed, showing local preview.');
          compiled = fallbackToLocal();
        }
      } else {
        try {
          compiled = await compileNeuroflowPipeline(nodes, edges, username);
        } catch (backendErr) {
          console.warn('Backend compile failed, falling back to local compiler', backendErr);
          compiled = fallbackToLocal();
        }
      }

      setCompiledData(compiled);
      setShowMonitor(true);
    } catch (error: any) {
      const message = error?.message || 'Failed to compile pipeline.';
      setValidationError(message);
    } finally {
      setIsSimulating(false);
    }
  };

  const handleSave = () => {
    const graphData = { nodes, edges };
    localStorage.setItem('neuroflow-graph', JSON.stringify(graphData));
    // Optional: add a temporary success message or toast
    alert("Pipeline saved successfully!");
  };

  const handleShare = () => {
    const graphData = {
      version: "1.0.2",
      timestamp: new Date().toISOString(),
      nodes,
      edges
    };
    const blob = new Blob([JSON.stringify(graphData, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `neuroflow-pipeline-${Date.now()}.json`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
  };

  const handleAIGraphGeneration = (newNodes: any[], newEdges: any[]) => {
    setNodes(newNodes);
    setEdges(newEdges);
  };

  return (
    <div className="flex flex-col h-screen w-screen overflow-hidden bg-[#09090b] text-white font-sans">
      {/* Top Bar */}
      <header className="h-14 border-b border-[#27272a] bg-[#121214] flex items-center justify-between px-6 z-10">
        <div className="flex items-center gap-3">
          <div className="bg-gradient-to-br from-blue-500 to-purple-600 w-8 h-8 rounded-lg flex items-center justify-center shadow-lg">
            <Activity size={20} className="text-white" />
          </div>
          <h1 className="font-bold text-lg tracking-tight text-gray-100">NeuroFlow <span className="text-xs text-blue-400 font-mono bg-blue-400/10 px-2 py-0.5 rounded border border-blue-400/20 ml-2">BETA</span></h1>
        </div>

        <div className="flex items-center gap-4">
          {validationError && (
            <div className="flex items-center gap-2 text-red-400 text-xs font-medium bg-red-900/10 border border-red-900/30 px-3 py-1 rounded animate-pulse">
              <AlertCircle size={14} /> {validationError}
            </div>
          )}

          <button
            onClick={handleSave}
            title="Save to local storage"
            className="p-2 hover:bg-[#27272a] rounded-md transition text-gray-400 hover:text-white"
          >
            <Save size={18} />
          </button>
          <button
            onClick={handleShare}
            title="Download JSON share file"
            className="p-2 hover:bg-[#27272a] rounded-md transition text-gray-400 hover:text-white"
          >
            <Share2 size={18} />
          </button>
          <div className="h-6 w-px bg-[#27272a]" />
          <button
            onClick={handleRunSimulation}
            disabled={isSimulating}
            className={`bg-emerald-600 hover:bg-emerald-500 text-white px-4 py-1.5 rounded-md flex items-center gap-2 text-sm font-semibold transition shadow-lg shadow-emerald-900/20 ${isSimulating ? 'opacity-70 cursor-wait' : ''}`}
          >
            {isSimulating ? <Terminal size={16} className="animate-spin" /> : <Play size={16} fill="currentColor" />}
            {isSimulating ? 'Compiling...' : 'Run Pipeline'}
          </button>
        </div>
      </header>

      {/* Main Workspace */}
      <div className="flex flex-1 overflow-hidden relative">
        <Sidebar />

        <div className="flex-1 relative" ref={reactFlowWrapper}>
          <ReactFlow
            nodes={nodes}
            edges={edges}
            onNodesChange={onNodesChange}
            onEdgesChange={onEdgesChange}
            onConnect={onConnect}
            onInit={setReactFlowInstance}
            onDrop={onDrop}
            onDragOver={onDragOver}
            onNodeClick={onNodeClick}
            onPaneClick={onPaneClick}
            nodeTypes={nodeTypes}
            proOptions={{ hideAttribution: true }}
            fitView
            snapToGrid
            snapGrid={[15, 15]}
          >
            <Background color="#27272a" variant={BackgroundVariant.Dots} gap={20} size={1} />
            <Controls className="!bg-[#18181b] !border-[#27272a] !fill-white" />
            <MiniMap
              nodeStrokeColor="#52525b"
              nodeColor="#18181b"
              maskColor="rgba(9, 9, 11, 0.6)"
              className="!bg-[#121214] !border-[#27272a] rounded-lg overflow-hidden"
            />
          </ReactFlow>

          {/* Overlay Grid / Decoration */}
          <div className="absolute top-4 left-4 pointer-events-none opacity-30">
            <div className="text-[10px] font-mono text-white">v1.0.2-alpha</div>
          </div>
        </div>

        {/* Properties Panel (Contextual) */}
        {selectedNode && (
          <PropertiesPanel
            selectedNode={selectedNode}
            onUpdateNode={updateNodeData}
            onDeleteNode={deleteNode}
            onClose={() => setSelectedNode(null)}
          />
        )}
      </div>

      <AIAssistant onGenerateGraph={handleAIGraphGeneration} />

      {/* The Backend Simulation Interface */}
      {compiledData && (
        <TrainingMonitor
          isOpen={showMonitor}
          onClose={() => setShowMonitor(false)}
          pipelineData={compiledData}
        />
      )}
    </div>
  );
};

export default function App() {
  return (
    <ReactFlowProvider>
      <Flow />
    </ReactFlowProvider>
  );
}
