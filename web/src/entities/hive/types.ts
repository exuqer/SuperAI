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
