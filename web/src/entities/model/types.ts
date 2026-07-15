/** Model entity types - typed API contracts. */

export interface CloudV2 {
  id: number;
  cloud_type:
    | 'character' | 'word_form' | 'lexeme' | 'scene' | 'concept_candidate' | 'concept'
    | 'morpheme_candidate' | 'morpheme' | 'morph_operator' | 'morph_pattern'
    | 'sentence_frame';
  canonical_name: string;
  mass: number;
  density: number;
  stability: number;
  base_activation: number;
  observation_count: number;
  metadata_json: string;
  created_at?: string;
  updated_at?: string;
}

export interface PlacementV2 {
  id: number;
  cloud_id: number;
  space_id: number;
  x: number;
  y: number;
  z: number | null;
  radius: number;
  local_activation: number;
  local_density: number;
  local_gravity: number;
  local_stability_modifier: number;
  metadata_json: string;
  created_at?: string;
  updated_at?: string;
}

export interface SpaceV2 {
  id: number;
  space_type:
    | 'global_field' | 'scene_space' | 'word_structure_space' | 'morphology_space'
    | 'sentence_frame_space' | 'concept_space' | 'hive_space' | 'hive_subspace';
  owner_cloud_id: number | null;
  parent_space_id: number | null;
  dimensionality?: number;
  random_seed: number;
  metadata_json?: string;
  created_at?: string;
}

export interface StructuralComponentV2 {
  id: number;
  parent_cloud_id: number;
  child_cloud_id: number;
  component_index: number;
  component_role: string;
  weight: number;
  local_x: number;
  local_y: number;
  local_z: number | null;
  metadata_json: string;
}

export interface SceneComponentV2 {
  id: number;
  placement_id: number;
  cloud_id: number;
  lexeme_cloud_id: number | null;
  token_index: number;
  grammatical_role: string;
  dependency_role: string | null;
  head_component_id: number | null;
  confidence: number;
  morphology_json: string;
}

export interface StatsV2 {
  clouds_total: number;
  clouds_by_type: Record<string, number>;
  spaces_total: number;
  spaces_by_type: Record<string, number>;
  placements_total: number;
  unique_word_forms: number;
  scene_components_total: number;
  structural_components_total: number;
  concepts_total: number;
  semantic_evidence_total: number;
  concept_fogs_total: number;
  concept_candidates_total: number;
  semantic_backfill_scenes_total: number;
}

export interface NormalizedSpaceV2 {
  space: SpaceV2;
  clouds: Record<string, CloudV2>;
  placements: PlacementV2[];
  stats: StatsV2;
}

export interface StructureV2 {
  cloud: CloudV2;
  structure_space: SpaceV2 | null;
  components: StructuralComponentV2[];
  clouds: Record<string, CloudV2>;
}

export interface SceneV2 {
  cloud_id: number;
  scene_space_id: number;
  sentence_text: string;
  canonical_text: string;
  observation_count: number;
  parser_version: string;
  components: SceneComponentV2[];
}

export interface TrainedModelSnapshotV2 {
  schema_version: number;
  stats: StatsV2;
  model: Record<string, unknown[]>;
}

// API Response wrappers
export type NormalizedSpaceResponse = NormalizedSpaceV2;
export type StructureResponse = StructureV2;
export type SceneResponse = SceneV2;
export type StatsResponse = StatsV2;
export type CloudResponse = CloudV2;
export type PlacementResponse = PlacementV2;
export type SpaceResponse = SpaceV2;
export type TrainedModelSnapshotResponse = TrainedModelSnapshotV2;

export interface PhysicsTickResponse {
  space_id: number;
  updates: Array<{
    placement_id: number;
    x: number;
    y: number;
    delta_x: number;
    delta_y: number;
  }>;
}

export interface InvariantCheckResponse {
  passed: boolean;
  violations: string[];
  checks: Record<string, { passed: boolean; violations: string[] }>;
}
