export type SemanticResult = {
  result_id: string;
  input_text: string;
  lang: string;
  tokens: string[];
  activated_concepts: ConceptSummary[];
  routes: AntRoute[];
  summary: string;
  response: string;
  sources: string[];
  session_id?: string | null;
  context_turns: ChatTurn[];
  semantic_vector: Record<string, unknown>;
  signal_trace: SignalStep[];
};

export type ConceptSummary = {
  uri: string;
  label: string;
  language: string;
  layer: number;
  score: number;
  sources: string[];
};

export type AntRoute = {
  ant_id: number;
  start: string;
  concepts: string[];
  total_score: number;
  steps: SignalStep[];
};

export type SignalStep = {
  ant_id?: number;
  step_index?: number;
  start: string;
  end: string;
  relation: string;
  layer: number;
  distance: number;
  remaining_strength?: number | null;
  edge_type: string;
  score: number;
};

export type ChatTurn = {
  role: string;
  text: string;
  result_id: string;
  concepts?: string[];
  created_at?: number;
};

export type ChatSession = {
  session_id: string;
  turns: ChatTurn[];
  turn_count: number;
  updated_at: number;
};

export type GraphNode = {
  id: string;
  uri: string;
  label: string;
  language: string;
  source: string;
  layer: number;
  metadata: Record<string, unknown>;
  concept_pheromone: number;
  suppression: number;
  degree: number;
  signal: { active: boolean; count: number };
};

export type GraphEdge = {
  id: string;
  start: string;
  end: string;
  relation: string;
  weight: number;
  source: string;
  surface_text?: string | null;
  layer: number;
  distance: number;
  edge_type: string;
  metadata: Record<string, unknown>;
  pheromone: number;
  route_stats: Record<string, unknown>;
  signal: { active: boolean; score: number };
};

export type GraphPayload = {
  nodes: GraphNode[];
  edges: GraphEdge[];
  stats: {
    nodes: number;
    edges: number;
    signal_nodes: number;
    signal_edges: number;
  };
};

export type AnalyzeResponse = {
  result: SemanticResult;
  graph: GraphPayload;
  trace_interpretation: TraceInterpretation;
};

export type TraceInterpretation = {
  summary: Record<string, unknown>;
  chains: Array<{
    ant_id: string;
    steps: SignalStep[];
    concept_chain: string[];
  }>;
  active_edge_ids: string[];
};

export type Job = {
  job_id: string;
  name: string;
  status: 'queued' | 'running' | 'completed' | 'failed';
  created_at: number;
  started_at?: number | null;
  finished_at?: number | null;
  result?: unknown;
  error?: string | null;
  traceback?: string | null;
};

export type ConceptDetail = {
  node: GraphNode;
  incoming: GraphEdge[];
  outgoing: GraphEdge[];
  aliases: string[];
};
