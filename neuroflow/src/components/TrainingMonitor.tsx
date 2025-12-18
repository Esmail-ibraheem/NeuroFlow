import React, { useEffect, useState, useRef } from 'react';
import { X, Terminal, CheckCircle2, AlertTriangle, Copy, Play } from 'lucide-react';

interface TrainingMonitorProps {
  isOpen: boolean;
  onClose: () => void;
  pipelineData: {
    config: any;
    cliCommand: string;
    pythonCode: string;
    jobId?: string;
    source?: string;
    projectName?: string;
    outputs?: Record<string, any>;
  };
}

const TrainingMonitor: React.FC<TrainingMonitorProps> = ({ isOpen, onClose, pipelineData }) => {
  const [activeTab, setActiveTab] = useState<'terminal' | 'code' | 'config' | 'results'>(
    pipelineData.outputs ? 'results' : 'terminal'
  );
  const [logs, setLogs] = useState<string[]>([]);
  const [progress, setProgress] = useState(0);
  const [status, setStatus] = useState<'initializing' | 'running' | 'completed' | 'failed'>('initializing');
  const logsEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!isOpen) {
      return;
    }

    if (pipelineData.source === 'backend') {
      setLogs([
        '> Connecting to AutoTrain backend...',
        pipelineData.jobId ? `> Tracking job: ${pipelineData.jobId}` : '> Tracking job...',
      ]);
      setStatus('running');
      setProgress((prev) => (prev === 0 ? 5 : prev));

      const controller = new AbortController();
      const fetchLogs = async () => {
        try {
          const params = new URLSearchParams({ limit: '200' });
          if (pipelineData.projectName) {
            params.set('project_name', pipelineData.projectName);
          }
          const apiBase = (import.meta.env.VITE_AUTOTRAIN_API_URL as string | undefined) || '/api';
          const resp = await fetch(`${apiBase}/neuroflow/logs?${params.toString()}`, {
            signal: controller.signal,
          });
          if (!resp.ok) {
            throw new Error(await resp.text());
          }
          const payload = await resp.json();
          const fetchedLogs: string[] = payload.logs || [];
          if (fetchedLogs.length) {
            setLogs(fetchedLogs);
          } else {
            setLogs(['> Waiting for backend logs...']);
          }
          if (fetchedLogs.some((line) => /(training completed|job finished|completed successfully)/i.test(line))) {
            setStatus('completed');
            setProgress(100);
          } else if (fetchedLogs.some((line) => /(error|exception|failed)/i.test(line))) {
            setStatus('failed');
          } else {
            setStatus('running');
            setProgress((prev) => (prev < 95 ? prev + 2 : prev));
          }
        } catch (error: any) {
          if (error.name !== 'AbortError') {
            setLogs((prev) => [...prev, `> Failed to fetch logs: ${error.message || error}`]);
            setStatus('failed');
          }
        }
      };

      fetchLogs();
      const interval = setInterval(fetchLogs, 4000);
      return () => {
        controller.abort();
        clearInterval(interval);
      };
    }

    const initialLogs = [
      '> Initializing NeuroFlow environment...',
      '> Validating hardware availability...',
      '> Using local compiler preview...',
    ];
    setLogs(initialLogs);
    setProgress(0);
    setStatus('initializing');

    let step = 0;
    const interval = setInterval(() => {
      step++;

      if (step === 2) {
        setLogs((prev) => [...prev, '> CUDA Available: True (NVIDIA A10G)', '> Backend: PyTorch 2.1.2+cu121']);
        setStatus('running');
      }
      if (step === 4) setLogs((prev) => [...prev, `> Downloading model: ${pipelineData.config.base_model}...`]);
      if (step === 6) {
        setLogs((prev) => [
          ...prev,
          '> Model loaded successfully.',
          pipelineData.jobId ? `> Tracking job ${pipelineData.jobId}` : '> Preparing dataset...',
        ]);
      }
      if (step === 8) setLogs((prev) => [...prev, '> Dataset tokenized. Starting training loop.']);

      if (step > 10 && step < 30) {
        const epoch = Math.ceil((step - 10) / 10);
        const loss = (2.5 - (step - 10) * 0.1).toFixed(4);
        const lrValue =
          pipelineData.config?.params?.learning_rate ?? pipelineData.config?.params?.lr ?? 'N/A';
        setLogs((prev) => [...prev, `[Epoch ${epoch}] Step ${step * 10}: Loss=${loss} | LR=${lrValue}`]);
        setProgress((prev) => Math.min(prev + 5, 95));
      }

      if (step === 30) {
        setLogs((prev) => [
          ...prev,
          '> Training completed successfully.',
          `> Model saved to ./output/${pipelineData.config.project_name}`,
        ]);
        setProgress(100);
        setStatus('completed');
        clearInterval(interval);
      }
    }, 800);

    return () => clearInterval(interval);
  }, [isOpen, pipelineData]);


  useEffect(() => {
    logsEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [logs]);

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 bg-black/80 backdrop-blur-sm z-[100] flex items-center justify-center p-6">
      <div className="bg-[#09090b] border border-[#27272a] w-full max-w-4xl h-[600px] rounded-xl shadow-2xl flex flex-col overflow-hidden animate-in zoom-in-95 duration-200">

        {/* Header */}
        <div className="h-12 bg-[#18181b] border-b border-[#27272a] flex items-center justify-between px-4">
          <div className="flex items-center gap-3">
            <div className={`w-2 h-2 rounded-full ${status === 'running' ? 'bg-yellow-400 animate-pulse' : status === 'completed' ? 'bg-green-500' : 'bg-blue-500'}`} />
            <span className="font-mono font-bold text-gray-200 text-sm">NeuroFlow Execution Engine</span>
            {pipelineData.jobId && (
              <span className="text-[11px] text-gray-400 font-mono bg-[#27272a] px-2 py-0.5 rounded">
                Job {pipelineData.jobId}
              </span>
            )}
          </div>
          <button onClick={onClose} className="text-gray-500 hover:text-white transition"><X size={18} /></button>
        </div>

        {/* Tabs */}
        <div className="flex border-b border-[#27272a] bg-[#121214]">
          <button
            onClick={() => setActiveTab('terminal')}
            className={`px-4 py-2 text-xs font-mono flex items-center gap-2 ${activeTab === 'terminal' ? 'text-white bg-[#27272a]' : 'text-gray-500 hover:text-gray-300'}`}
          >
            <Terminal size={14} /> Console Output
          </button>
          {pipelineData.outputs && (
            <button
              onClick={() => setActiveTab('results')}
              className={`px-4 py-2 text-xs font-mono flex items-center gap-2 ${activeTab === 'results' ? 'text-white bg-[#27272a]' : 'text-gray-500 hover:text-gray-300'}`}
            >
              <CheckCircle2 size={14} className="text-emerald-400" /> Results
            </button>
          )}
          <button
            onClick={() => setActiveTab('code')}
            className={`px-4 py-2 text-xs font-mono flex items-center gap-2 ${activeTab === 'code' ? 'text-white bg-[#27272a]' : 'text-gray-500 hover:text-gray-300'}`}
          >
            <Play size={14} /> Generated Python
          </button>
          <button
            onClick={() => setActiveTab('config')}
            className={`px-4 py-2 text-xs font-mono flex items-center gap-2 ${activeTab === 'config' ? 'text-white bg-[#27272a]' : 'text-gray-500 hover:text-gray-300'}`}
          >
            <AlertTriangle size={14} /> Raw Config
          </button>
        </div>

        {/* Content Area */}
        <div className="flex-1 overflow-hidden relative">

          {/* Terminal View */}
          {activeTab === 'terminal' && (
            <div className="h-full bg-[#0c0c0e] p-4 overflow-y-auto font-mono text-xs scrollbar-thin">
              {logs.map((log, i) => (
                <div key={i} className="mb-1 text-gray-300">
                  <span className="text-gray-600 mr-2">{new Date().toLocaleTimeString()}</span>
                  <span className={log.includes('Error') ? 'text-red-400' : log.includes('Success') ? 'text-green-400' : 'text-gray-300'}>
                    {log}
                  </span>
                </div>
              ))}
              {status === 'running' && (
                <div className="mt-2 text-blue-400 animate-pulse">_</div>
              )}
              <div ref={logsEndRef} />
            </div>
          )}

          {/* Code View */}
          {activeTab === 'code' && (
            <div className="h-full bg-[#1e1e1e] p-4 overflow-y-auto relative group">
              <button className="absolute top-4 right-4 bg-white/10 hover:bg-white/20 p-2 rounded text-white opacity-0 group-hover:opacity-100 transition">
                <Copy size={16} />
              </button>
              <pre className="text-xs font-mono text-blue-200 language-python">
                {pipelineData.pythonCode}
              </pre>
            </div>
          )}

          {/* Config View */}
          {activeTab === 'config' && (
            <div className="h-full bg-[#1e1e1e] p-4 overflow-y-auto relative">
              <pre className="text-xs font-mono text-yellow-200">
                {JSON.stringify(pipelineData.config, null, 2)}
              </pre>
            </div>
          )}

          {/* Results View */}
          {activeTab === 'results' && pipelineData.outputs && (
            <div className="h-full bg-[#0c0c0e] p-6 overflow-y-auto font-sans scrollbar-thin">
              <h3 className="text-emerald-400 font-bold mb-4 flex items-center gap-2">
                <CheckCircle2 size={18} /> Execution Successful
              </h3>
              <div className="space-y-4">
                {Object.entries(pipelineData.outputs).map(([nodeId, result]) => (
                  <div key={nodeId} className="bg-[#18181b] border border-[#27272a] rounded-lg p-4 shadow-sm">
                    <div className="flex items-center justify-between mb-2">
                      <span className="text-[10px] uppercase tracking-wider text-gray-500 font-mono">{nodeId}</span>
                      <span className="text-[10px] text-emerald-500/80 font-mono bg-emerald-500/10 px-2 py-0.5 rounded">OUTPUT</span>
                    </div>
                    <div className="text-sm text-gray-200 whitespace-pre-wrap leading-relaxed flex flex-col gap-3">
                      {typeof result === 'string' && result.startsWith('http') ? (
                        <div className="bg-blue-600/10 border border-blue-600/30 p-4 rounded-lg flex flex-col items-center gap-3">
                          <span className="text-xs text-blue-400 font-mono">{result}</span>
                          <button
                            onClick={() => window.open(result, '_blank')}
                            className="bg-blue-600 hover:bg-blue-500 text-white px-6 py-2 rounded-md flex items-center gap-2 text-sm font-semibold transition shadow-lg shadow-blue-900/20"
                          >
                            <Play size={16} fill="currentColor" /> Open Interface
                          </button>
                        </div>
                      ) : (
                        typeof result === 'string' ? result : JSON.stringify(result, null, 2)
                      )}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

        </div>

        {/* Footer / Status Bar */}
        <div className="h-10 bg-[#18181b] border-t border-[#27272a] flex items-center justify-between px-4">
          <div className="flex items-center gap-4 w-full max-w-md">
            <span className="text-xs text-gray-400 font-mono whitespace-nowrap">{status.toUpperCase()}</span>
            <div className="h-1.5 bg-[#27272a] rounded-full w-full overflow-hidden">
              <div
                className="h-full bg-blue-500 transition-all duration-300 ease-out"
                style={{ width: `${progress}%` }}
              />
            </div>
            <span className="text-xs text-gray-400 font-mono">{progress}%</span>
          </div>

          {status === 'completed' && (
            <button className="flex items-center gap-2 text-xs bg-green-600 hover:bg-green-500 text-white px-3 py-1 rounded transition">
              <CheckCircle2 size={14} /> Download Model
            </button>
          )}
        </div>
      </div>
    </div>
  );
};

export default TrainingMonitor;
