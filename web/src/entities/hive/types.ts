/** Hive entity types - typed API contracts. */

export interface HiveV2 {
  id: string;
  space_id: number;
  query_text: string;
  query_json: Record<string, unknown>;
  max_cells: number;
  created_at: string;
  updated_at: string;
}

export interface HiveCellV2 {
  id: string;
  hive_id: string;
  dominant_cloud_id: number;
  hive_placement_id: number;
  source_cloud_id: number;
  source_placement_id: number | null;
  source_space_id: number | null;
  source_scene_cloud_id: number | null;
  stored_strength: number;
  retention: number;
  local_activation: number;
  component_class: string;
  metadata: Record<string, unknown>;
  created_at: string;
  updated_at: string;
  // Computed/display fields
  label?: string;
  x?: number;
  y?: number;
  gravity?: number;
  components?: HiveCellComponentV2[];
  subspaces?: HiveSubspaceV2[];
}

export interface HiveSubspaceV2 {
  id: number;
  hive_id: string;
  parent_cell_id: string | null;
  space_id: number;
  subspace_type: 'lexeme' | 'word_form' | 'morphology' | 'characters' | string;
  depth: number;
  capacity: number;
  status: string;
  expansion_reason: string;
}

export interface HiveCellComponentV2 {
  id: number;
  cell_id: string;
  cloud_id: number;
  composition_share: number;
  local_activation: number;
  role: string;
  effective_strength: number;
  component_class: string;
  source_cloud_id: number;
  source_placement_id: number | null;
  source_space_id: number | null;
  provenance: Record<string, unknown>;
  // Joined fields
  canonical_name?: string;
  cloud_type?: string;
}

export interface HiveMessageV2 {
  id: string;
  hive_id: string;
  turn_index: number;
  text: string;
  parsed_json: Record<string, unknown>;
  created_at: string;
  role: 'user' | 'assistant';
}

export interface HiveQueryDecisionV2 {
  decision: string;
  external_search_required: boolean;
  reasons: string[];
  matches: Array<{ cell_id: string; local_support: number }>;
  unresolved_components?: Array<Record<string, unknown>>;
  local_anchors?: Array<Record<string, unknown>>;
  external_request?: Record<string, unknown> | null;
}

export interface HiveResonanceEventV2 {
  id: string;
  hive_id: string;
  message_id: string;
  cell_id: string;
  component_cloud_id: number | null;
  reason: string;
  payload_json: string;
  created_at: string;
}

export interface HiveCellMatchV2 {
  id: number;
  decision_id: string;
  cell_id: string;
  component_id: string;
  match_type: string;
  local_support: number;
  metadata_json: string;
}

export interface HiveReasoningRunV2 {
  id: string;
  hive_id: string;
  status: string;
  reasoning_steps: number;
  completed_steps: number;
  query_json: string;
  config_json: string;
  random_seed: number;
  stop_reason: string | null;
  initial_state_hash: string | null;
  final_state_hash: string | null;
  created_at: string;
  completed_at: string | null;
}

export interface HiveReasoningSnapshotV2 {
  id: string;
  run_id: string;
  hive_id: string;
  step: number;
  phase: string;
  state_hash: string;
  state_json: string;
  delta_json: string;
  clusters_json: string;
  events_json: string;
  created_at: string;
}

export interface HiveReasoningEventV2 {
  id: string;
  run_id: string;
  hive_id: string;
  step: number;
  phase: string;
  event_type: string;
  placement_id: number | null;
  payload_json: string;
  created_at: string;
}

export interface HiveResonanceClusterV2 {
  id: string;
  run_id: string;
  hive_id: string;
  reasoning_step: number;
  member_placement_ids_json: string;
  dominant_cloud_ids_json: string;
  cohesion: number;
  total_energy: number;
  average_gravity: number;
  query_relevance: number;
  status: string;
  created_at: string;
}

export interface HiveAnalyticsRunV2 {
  id: string;
  hive_id: string;
  status: string;
  reasoning_steps: number;
  completed_steps: number;
  stop_reason: string | null;
  random_seed: number;
  created_at: string;
  completed_at: string | null;
  query: { terms?: string[]; roles?: string[]; cloud_ids?: number[] };
  config: Record<string, unknown>;
}

export interface HiveAnalyticsNodeV2 {
  placement_id: number;
  cell_id: string | null;
  cloud_id: number;
  node_type: string;
  label: string;
  local_activation: number;
  local_gravity: number;
  retention: number;
  energy: number;
  eviction_status: string;
}

export interface HiveAnswerCandidateV2 {
  placement_id: number;
  cell_id: string | null;
  scene_cloud_id: number;
  scene_label: string;
  answer: string | null;
  matched_components: Array<{ term: string; role: string; label: string }>;
  answer_components: Array<{ answer: string; role: string; question_term: string }>;
  semantic_score: number;
  dynamic_score: number;
  viability: number;
  candidate_score: number;
  eviction_status: string;
  explanation: string;
}

export interface HiveAnalyticsSnapshotV2 {
  step: number;
  phase: string;
  created_at: string;
  temperature: number;
  metrics: {
    average_activation: number;
    average_retention: number;
    total_energy: number;
    active_nodes: number;
    weakening_nodes: number;
    evicted_nodes: number;
  };
  nodes: HiveAnalyticsNodeV2[];
  candidates: HiveAnswerCandidateV2[];
  delta: Record<string, unknown>;
  events: Array<Record<string, unknown>>;
}

export interface HiveAnalyticsRunResultV2 {
  run: HiveAnalyticsRunV2;
  query_components: Array<{ term: string; role: string; word_form_cloud_id: number | null }>;
  snapshots: HiveAnalyticsSnapshotV2[];
  events: Array<Record<string, unknown>>;
  clusters: Array<Record<string, unknown>>;
}

export interface HiveAnalyticsCurrentV2 {
  query_components: Array<{ term: string; role: string; word_form_cloud_id: number | null }>;
  snapshot: HiveAnalyticsSnapshotV2;
  updated_at: string | null;
}

export interface HiveAnalyticsResponse {
  hive_id: string;
  current: HiveAnalyticsCurrentV2;
  runs: HiveAnalyticsRunV2[];
  primary: HiveAnalyticsRunResultV2 | null;
  comparison: HiveAnalyticsRunResultV2 | null;
}
