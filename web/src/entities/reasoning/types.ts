/** Reasoning entity types - typed API contracts. */

export interface ReasoningRunV2 {
  id: string;
  hive_id: string;
  status: string;
  reasoning_steps: number;
  completed_steps: number;
  query_json: Record<string, unknown>;
  config_json: Record<string, unknown>;
  random_seed: number;
  stop_reason: string | null;
  initial_state_hash: string | null;
  final_state_hash: string | null;
  created_at: string;
  completed_at: string | null;
}

export interface ReasoningSnapshotV2 {
  id: string;
  run_id: string;
  hive_id: string;
  step: number;
  phase: string;
  state_hash: string;
  state_json: {
    nodes: Record<string, ReasoningNodeState>;
    step: number;
    clusters: ReasoningClusterV2[];
  };
  delta_json: Record<string, unknown>;
  clusters_json: ReasoningClusterV2[];
  events_json: ReasoningEventV2[];
  created_at: string;
}

export interface ReasoningNodeState {
  placement_id: number;
  cloud_id: number;
  x: number;
  y: number;
  activation: number;
  gravity: number;
  stored_strength: number;
  retention: number;
  energy: number;
  phase: number;
  frequency: number;
  temperature_response: number;
  age_steps: number;
  activation_count: number;
  last_activated_step: number;
  weakening_steps: number;
  eviction_status: string;
  is_query: boolean;
}

export interface ReasoningClusterV2 {
  id: string;
  member_placement_ids: number[];
  dominant_cloud_ids: number[];
  cohesion: number;
  total_energy: number;
  average_gravity: number;
  query_relevance: number;
  status: string;
}

export interface ReasoningEventV2 {
  id: string;
  run_id: string;
  hive_id: string;
  step: number;
  phase: string;
  event_type: string;
  placement_id: number | null;
  payload_json: Record<string, unknown>;
  created_at: string;
}

export interface ReasoningDiffV2 {
  run_id: string;
  from_step: number;
  to_step: number;
  added_nodes: string[];
  removed_nodes: string[];
  changed_nodes: string[];
  clusters_delta: number;
}

export interface VibrationConfigV2 {
  reasoning_steps?: number;
  temperature?: number;
  gravity_threshold?: number;
  cluster_threshold?: number;
  random_seed?: number;
}

export interface QueryActivationV2 {
  word_form_cloud_ids: number[];
  normalized_forms: string[];
  expected_roles: string[];
}

export interface ReasoningExportV2 {
  run: ReasoningRunV2;
  snapshots: ReasoningSnapshotV2[];
  events: ReasoningEventV2[];
  hive?: Record<string, unknown>;
  export_mode: 'current' | 'snapshot' | 'trace' | 'initial';
  detail: string;
}