import { Edge, Node } from 'reactflow';
import { NodeData } from '../types';

const API_BASE = import.meta.env.VITE_AUTOTRAIN_API_URL || '/api';

export interface BackendPipelinePayload {
  config: Record<string, any>;
  cliCommand: string;
  pythonCode: string;
  jobId?: string;
  projectName?: string;
  task?: string;
  backend?: string;
  source: 'backend' | 'local';
  outputs?: Record<string, any>;
}

const ensureResponse = async (response: Response) => {
  if (!response.ok) {
    const payload = await response.json().catch(() => ({}));
    const message = payload.detail || payload.error || response.statusText;
    throw new Error(typeof message === 'string' ? message : 'Backend request failed');
  }
  return response.json();
};

export const compileNeuroflowPipeline = async (
  nodes: Node<NodeData>[],
  edges: Edge[],
  username?: string
): Promise<BackendPipelinePayload> => {
  const response = await fetch(`${API_BASE}/neuroflow/compile`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ nodes, edges, username }),
  });
  const data = await ensureResponse(response);
  return normalizeBackendPayload(data, true);
};

export const runNeuroflowPipeline = async (
  nodes: Node<NodeData>[],
  edges: Edge[],
  token?: string,
  username?: string
): Promise<BackendPipelinePayload> => {
  const headers: Record<string, string> = {
    'Content-Type': 'application/json',
  };
  if (token) {
    headers.Authorization = `Bearer ${token}`;
  }

  const response = await fetch(`${API_BASE}/neuroflow/run`, {
    method: 'POST',
    headers,
    body: JSON.stringify({ nodes, edges, username }),
  });
  const data = await ensureResponse(response);
  return normalizeBackendPayload(data, true);
};

export const uploadNeuroflowFile = async (file: File): Promise<{ filename: string; path: string; success: boolean }> => {
  const formData = new FormData();
  formData.append('file', file);

  const response = await fetch(`${API_BASE}/neuroflow/upload`, {
    method: 'POST',
    body: formData,
  });
  return ensureResponse(response);
};

const normalizeBackendPayload = (raw: any, isBackend = false): BackendPipelinePayload => {
  return {
    config: raw.config || {},
    cliCommand: raw.cli_command || raw.cliCommand || '',
    pythonCode: raw.python_code || raw.pythonCode || '',
    jobId: raw.job_id || raw.jobId,
    projectName: raw.project_name || raw.projectName,
    task: raw.task,
    backend: raw.backend,
    source: isBackend ? 'backend' : 'local',
    outputs: raw.outputs,
  };
};
