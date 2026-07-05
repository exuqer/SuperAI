import type {
  AnalyzeResponse,
  DecodeResponse,
  ChatSession,
  ConceptDetail,
  GraphPayload,
  Job,
  UnderstandingResponse,
} from './types';

async function request<T>(path: string, options: RequestInit = {}): Promise<T> {
  const response = await fetch(path, {
    headers: {
      'Content-Type': 'application/json',
      ...(options.headers ?? {}),
    },
    ...options,
  });
  if (!response.ok) {
    const detail = await response.text();
    throw new Error(`${response.status} ${response.statusText}: ${detail}`);
  }
  return (await response.json()) as T;
}

function query(params: Record<string, unknown>): string {
  const values = new URLSearchParams();
  Object.entries(params).forEach(([key, value]) => {
    if (value !== undefined && value !== null && value !== '') {
      values.set(key, String(value));
    }
  });
  const text = values.toString();
  return text ? `?${text}` : '';
}

export const api = {
  getHealth: () => request<Record<string, unknown>>('/api/health'),
  getConfig: () => request<Record<string, unknown>>('/api/config'),
  analyze: (payload: Record<string, unknown>) =>
    request<AnalyzeResponse>('/api/analyze', { method: 'POST', body: JSON.stringify(payload) }),
  understand: (payload: Record<string, unknown>) =>
    request<UnderstandingResponse>('/api/understand', { method: 'POST', body: JSON.stringify(payload) }),
  decode: (payload: Record<string, unknown>) =>
    request<DecodeResponse>('/api/decode', { method: 'POST', body: JSON.stringify(payload) }),
  chatMessage: (payload: Record<string, unknown>) =>
    request<AnalyzeResponse>('/api/chat/message', { method: 'POST', body: JSON.stringify(payload) }),
  getSessions: () => request<ChatSession[]>('/api/chat/sessions'),
  resetSession: (sessionId: string) => request(`/api/chat/sessions/${encodeURIComponent(sessionId)}`, { method: 'DELETE' }),
  sendFeedback: (payload: Record<string, unknown>) =>
    request<Record<string, unknown>>('/api/feedback', { method: 'POST', body: JSON.stringify(payload) }),
  interpretVector: (semanticVector: unknown) =>
    request<{ response: string }>('/api/vector/interpret', {
      method: 'POST',
      body: JSON.stringify({ semantic_vector: semanticVector }),
    }),
  getMemorySummary: () => request<Record<string, unknown>>('/api/memory/summary'),
  getMemoryResults: () => request<unknown[]>('/api/memory/results'),
  getMemoryCollections: () => request<Record<string, unknown>>('/api/memory/collections'),
  getConcepts: (params: Record<string, unknown>) =>
    request<unknown[]>(`/api/concepts${query(params)}`),
  getConceptDetail: (uri: string, resultId?: string | null) =>
    request<ConceptDetail>(`/api/concepts/detail${query({ uri, result_id: resultId })}`),
  getGraph: (params: Record<string, unknown>) => request<GraphPayload>(`/api/graph${query(params)}`),
  getJobs: () => request<Job[]>('/api/jobs'),
  getJob: (jobId: string) => request<Job>(`/api/jobs/${encodeURIComponent(jobId)}`),
  train: (payload: Record<string, unknown>) =>
    request<Job>('/api/training/train', { method: 'POST', body: JSON.stringify(payload) }),
  learn: (payload: Record<string, unknown>) =>
    request<Job>('/api/training/learn', { method: 'POST', body: JSON.stringify(payload) }),
  learnDialogues: (payload: Record<string, unknown>) =>
    request<Job>('/api/training/learn-dialogues', { method: 'POST', body: JSON.stringify(payload) }),
  simpleTrain: (payload: Record<string, unknown>) =>
    request<Job>('/api/training/simple', { method: 'POST', body: JSON.stringify(payload) }),
  eval: (payload: Record<string, unknown>) =>
    request<Job>('/api/eval', { method: 'POST', body: JSON.stringify(payload) }),
  dream: (payload: Record<string, unknown>) =>
    request<Job>('/api/system/dream', { method: 'POST', body: JSON.stringify(payload) }),
  bootstrap: (payload: Record<string, unknown>) =>
    request<Job>('/api/system/bootstrap', { method: 'POST', body: JSON.stringify(payload) }),
  resetNetwork: (payload: Record<string, unknown>) =>
    request<Job>('/api/system/reset-network', { method: 'POST', body: JSON.stringify(payload) }),
  downloadSpc: (payload: Record<string, unknown>) =>
    request<Job>('/api/datasets/spc/download', { method: 'POST', body: JSON.stringify(payload) }),
  exportCheckpoint: (payload: Record<string, unknown>) =>
    request<Job>('/api/system/export', { method: 'POST', body: JSON.stringify(payload) }),
};
