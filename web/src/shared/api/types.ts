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
  layers?: number[];
  active_layers?: number[];
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
  from_layer?: number | null;
  to_layer?: number | null;
  context_plane?: string | null;
  layer_pheromone?: number;
  projection_shift?: number;
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
  layers: number[];
  active_layers: number[];
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
  from_layer?: number | null;
  to_layer?: number | null;
  context_plane?: string | null;
  distance: number;
  edge_type: string;
  metadata: Record<string, unknown>;
  pheromone: number;
  layer_pheromone?: number;
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

export type UnderstandingToken = {
  raw_token: string;
  lemma: string;
  search_token: string;
  concept_uri: string | null;
  match_status:
    | 'found_as_alias'
    | 'found_as_lemma'
    | 'found_as_raw'
    | 'candidate'
    | 'partial_root_match'
    | 'edit_distance_match'
    | 'ignored_stop_word';
  is_stop_word: boolean;
  morphology: {
    POS: string | null;
    case: string | null;
    number: string | null;
    gender: string | null;
    tense: string | null;
    person: string | null;
  };
};

export type UnderstandingSummary = {
  total_tokens: number;
  working_tokens: number;
  stop_words: number;
  matched: number;
  candidates: number;
  partial_root_matches: number;
  edit_distance_matches: number;
  search_tokens: string[];
};

export type UnderstandingResponse = {
  input_text: string;
  lang: string;
  session_id?: string | null;
  turn_id?: string | null;
  tokens: UnderstandingToken[];
  summary: UnderstandingSummary;
};

export type DecodeToken = {
  input_token: string;
  normalized_token: string;
  role: 'subject' | 'verb' | 'object' | 'instrument' | 'location' | 'complement' | 'modifier';
  surface: string;
  concept_uri: string | null;
  transform_status: 'inflected' | 'surface' | 'fallback';
  morphology: {
    POS: string | null;
    case: string | null;
    number: string | null;
    gender: string | null;
    tense: string | null;
    person: string | null;
  };
};

export type DecodeSummary = {
  total_tokens: number;
  used_tokens: number;
  objects: number;
  fallbacks: number;
};

export type DecodeResponse = {
  input_text: string;
  input_tokens: string[];
  lang: string;
  sentence: string;
  pattern: 's' | 'svo' | 'svoc' | 'svoi' | 'svm' | 'empty';
  session_id?: string | null;
  turn_id?: string | null;
  tokens: DecodeToken[];
  summary: DecodeSummary;
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
