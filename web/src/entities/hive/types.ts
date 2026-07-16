/** Hive entity types - typed API contracts. */

export interface HiveV2 {
  id: string;
  space_id: number;
  query_text: string;
  query_json: Record<string, unknown>;
  max_cells: number;
  created_at: string;
  updated_at: string;
  capacity?: number | Record<string, number>;
  energy?: Record<string, number>;
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
  projection?: HiveProjectionV2;
}

/** A filtered, bounded view into the single hive projection of the global field. */
export interface HiveProjectionV2 {
  space_type: 'hive' | 'scene' | 'lexeme' | 'morphology' | 'word_form' | 'characters' | string;
  scope: 'bounded_field_projection';
  source_space_id: number | null;
  parent_projection_id: number | null;
  parent_node_id: string | null;
  depth: number;
  capacity: number;
  filter?: { cell_id: string | null; resolution: string };
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

export interface HiveInspectionProjectionV2 {
  id: string;
  projection_type: string;
  source_cell_id: string | null;
  subspace_id: number | null;
  status: string;
  forms: Array<Record<string, unknown>>;
}

export interface QuerySceneCandidateV2 {
  id: string;
  lemma: string;
  surface: string;
  target_role: string;
  status: string;
  sources: string[];
  primary_source_id?: string;
  grammatical_features?: Record<string, string>;
  form_provenance?: {
    source_type: string;
    scene_id?: string;
    scene_text?: string;
    observed_surface?: string;
    generated: boolean;
  };
  scores: Record<string, number>;
  answer_mode?: 'direct' | 'explanation' | string;
  cell_id?: string | null;
}

export interface DialogueContextV2 {
  location?: Record<string, unknown> | null;
  destination?: Record<string, unknown> | null;
  source?: Record<string, unknown> | null;
}

export interface ContextResolutionV2 {
  status: 'NOT_APPLICABLE' | 'RESOLVED' | 'UNRESOLVED_CONTEXT' | string;
  referential?: string | null;
  role?: string | null;
  value?: Record<string, unknown> | null;
}

export interface QuerySceneV2 {
  id: string;
  type: 'query_scene';
  status: string;
  requested_role: string | null;
  slots: Array<Record<string, unknown>>;
}

export type QueryMessageMode = 'NEW_QUERY' | 'LOCAL_RESONANCE' | 'FOLLOW_UP' | 'CORRECTION';

export interface HiveLocalResonanceV2 {
  latest_probe_id: string;
  latest_surface: string;
  status: string;
  probe_text?: string;
  matched_form?: string | null;
  matched_lexeme?: string | null;
}

export interface HiveResonanceSessionV2 {
  id: string;
  input: string;
  status: string;
  tick: number;
  max_ticks: number;
  temperature: number;
  energy_budget: number;
  stability: number;
  completion_reason?: string | null;
  lexical_candidates: Array<Record<string, unknown>>;
  active_concepts: Array<Record<string, unknown>>;
  suppressed_concepts: Array<Record<string, unknown>>;
  dominant_configuration?: Record<string, unknown> | null;
  snapshots: Array<Record<string, unknown>>;
}

export interface QueryWorkingHiveV2 {
  query_frame: Record<string, unknown>;
  query_scene: QuerySceneV2;
  memory_scenes: Array<Record<string, unknown>>;
  candidates: QuerySceneCandidateV2[];
  vibration: { current_step: number; status: string; history: Array<Record<string, unknown>> };
  answer: { answer_mode: string; surface_answer: string; confidence: number };
  dialogue_context?: DialogueContextV2;
  context_resolution?: ContextResolutionV2;
  retrieval_scope?: Record<string, number>;
  semantic_total?: number;
  gravity?: number;
  decision_score?: number;
}

export interface DynamicsNodeStateV2 {
  cell_id: string;
  label?: string;
  node_type: string;
  role?: string | null;
  position: { x: number; y: number };
  previous_position?: { x: number; y: number };
  velocity: { x: number; y: number };
  acceleration: { x: number; y: number };
  mass: { global: number; local: number };
  activation: number;
  retention: number;
  resonance: number;
  gravity: number;
  local_gravity?: number;
  energy: number;
  net_force: { x: number; y: number; magnitude: number };
  distance_to_core: number;
  distance_to_target: number;
  eviction_score: number;
  eviction_status: string;
  zone: string;
  force_breakdown: Array<Record<string, unknown>>;
  trajectory: Array<{ step: number; x: number; y: number }>;
}

export interface DynamicsStateV2 {
  version: number;
  step: number;
  status: string;
  temperature: { status?: string; initial: number | null; current: number | null; minimum: number; maximum: number; cooling_rate: number; state?: string; history: Array<{ step: number; value: number }> };
  capacity_pressure: number;
  center_of_mass: { x: number; y: number };
  zones: Record<string, number>;
  anchors: Array<Record<string, unknown>>;
  nodes: DynamicsNodeStateV2[];
  history: Array<Record<string, unknown>>;
  eviction_history: Array<Record<string, unknown>>;
  random_seed: number;
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

export interface HiveSnapshotWordV2 {
  id: string;
  node_type: 'word' | 'concept' | string;
  lemma_cloud_id: number | null;
  word_form_cloud_id?: number | null;
  lemma: string;
  surface_forms: Array<{ surface: string; word_form_cloud_id: number; count: number }>;
  global: { mass: number; density: number; stability: number; observation_count: number };
  local: { activation: number; gravity: number; stored_strength: number; retention: number; energy: number };
  position: { base_x: number; base_y: number; render_x: number; render_y: number };
  roles: string[];
  scene_support_count: number;
  contributions: Array<{ scene_id: string; role: string; surface: string; scene_activation: number; scene_gravity: number; stored_strength: number }>;
  resonance: { active: boolean; displacement: [number, number]; velocity: [number, number]; emitted_energy: number; received_energy: number; support: number; suppression: number; temperature_noise: number };
}

export interface HiveSnapshotSceneV2 {
  id: string;
  cloud_id: number;
  text: string;
  source: string;
  cell_id: string;
  position: { x: number; y: number };
  physics: { mass: number; local_activation: number; local_gravity: number; stored_strength: number; retention: number; energy: number };
  status: Record<string, string>;
  roles: Record<string, { lemma: string; surface: string }>;
  match: { total_score: number; matched_roles: string[]; mismatched_roles: string[]; selection_reason: string };
}

export interface HiveSnapshotV2 {
  schema_version: number;
  hive: { id: string; status: string; capacity: number; occupied_cells: number; temperature: number; reasoning_step: number; energy: number };
  summary: { scene_count: number; word_count: number; concept_count: number; active_word_count: number; query_anchor_count: number; candidate_scene_count: number; rejected_scene_count: number; resonance_status: string };
  scenes: HiveSnapshotSceneV2[];
  words: HiveSnapshotWordV2[];
  query_overlay: Record<string, any>;
  resonance: Record<string, any>;
  timeline: Array<Record<string, any>>;
  field: { center_of_mass: { x: number; y: number }; zones: Record<string, number> };
  diagnostics: { warnings: Array<{ code: string; message: string }>; counts: Record<string, number> };
}
