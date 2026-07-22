/** Public client contracts for the role-free SuperAI V3.0 graph API. */

export type QueryMode = 'NEW_QUERY' | 'FOLLOW_UP' | 'CORRECTION';
export type RetrievalScope = 'LOCAL_ONLY' | 'LOCAL_THEN_GLOBAL' | 'GLOBAL_ONLY';
export type GraphStatus = 'READY' | 'AMBIGUOUS' | 'INCOMPLETE' | 'CONFLICTED';
export type AnswerStatus =
  | 'RESOLVED'
  | 'PARTIALLY_RESOLVED'
  | 'UNRESOLVED'
  | 'AMBIGUOUS'
  | 'CONFLICTED'
  | 'BUILD_FAILED';

export interface ModelVersions {
  event_schema_version?: string;
  slot_model_version?: string;
  construction_model_version?: string;
  semantic_cluster_version?: string;
  query_graph_version?: string;
  generation_version?: string;
  migration_version?: string;
}

export interface MentionComponent {
  component_id: string;
  lemma: string;
  surface: string;
  token_index: number;
  attachment_signature: Record<string, number>;
  required: boolean;
}

export interface MentionNode {
  node_id: string;
  node_type: 'MENTION';
  head: {
    lemma: string;
    surface: string;
  };
  surface: string;
  token_start: number;
  token_end: number;
  token_indices: number[];
  features: Record<string, unknown>;
  components: MentionComponent[];
  preposition: string;
  entity_id: string | null;
  semantic_cluster_ids: string[];
  qualified_key: string;
}

export interface PredicateNode {
  lemma: string;
  surface: string;
  concept_id: string;
  token_index: number;
  features: Record<string, unknown>;
}

export interface GapNode {
  node_id: string;
  node_type: 'GAP';
  gap_kind:
    | 'EVENT_ATTACHMENT'
    | 'NODE_COMPONENT'
    | 'RELATION_VALUE'
    | 'EVENT_PROPERTY'
    | 'BOOLEAN_RESULT'
    | 'QUANTITY_VALUE'
    | 'WHOLE_EVENT';
  surface: string;
  token_indices: number[];
  attached_to_node_id: string | null;
  required?: boolean;
  coordination_group_id?: string | null;
  question_signature: Record<string, number>;
  compatible_slot_hypotheses: Record<string, number>;
}

export interface QueryGraph {
  query_graph_id: string;
  question_operators?: Array<Record<string, unknown>>;
  event_pattern: {
    predicate: PredicateNode | null;
    known_nodes: MentionNode[];
    gap_node: GapNode;
    target_gaps?: GapNode[];
    target_gap?: GapNode;
    required_edges: Array<Record<string, unknown>>;
  };
  exclusions: Array<Record<string, unknown>>;
  status: GraphStatus;
  continuation_of: string | null;
  construction_ids: string[];
  trace: Record<string, unknown>;
  versions: ModelVersions;
}

export interface CandidateBinding {
  binding_id: string;
  query_graph_id: string;
  event_id: string;
  gap_node_id: string;
  resolved_node_id: string;
  resolved_concept_id: string;
  resolved_lemma: string;
  resolved_surface: string;
  resolved_features: Record<string, unknown>;
  scores: {
    structural: number;
    signature: number;
    evidence: number;
    total: number;
  };
  status: 'CANDIDATE' | 'ACCEPTED' | 'REJECTED' | 'SELECTED';
  failed_constraint: string | null;
  evidence: Array<Record<string, unknown>>;
}

export interface GraphAnswer {
  status: AnswerStatus;
  short_answer: string | null;
  full_answer: string | null;
  surface: string | null;
  chat_text?: string;
  selected_bindings?: CandidateBinding[];
  provenance?: {
    source_event_ids: string[];
    independent_source_count: number;
  };
  validation: {
    valid: boolean;
    reason?: string;
    failures?: string[];
    checks?: string[];
  };
  versions?: ModelVersions;
}

export interface GapSwarmRun {
  id: string;
  gap_id: string;
  status: string;
  termination_reason: string;
  retrieval_mode: 'SWARM_DIMENSIONAL' | 'SWARM_MIXED' | 'INDEX_FALLBACK' | 'DIRECT_EVENT_LOOKUP';
  fallback_reason?: string;
  events_considered: number;
  events_returned: number;
  missions: Array<{
    bee_id: string;
    bee_type: string;
    mission_type: string;
    seed?: Record<string, unknown>;
    visited_universes: string[];
    candidate_event_ids?: string[];
    successful: boolean;
    termination_reason?: string;
  }>;
  nectar_packets?: Array<{
    packet_id: string;
    source_universe: string;
    target_universe: string;
    event_ids: string[];
    dimension_ids: string[];
    evidence_weight: number;
  }>;
}

export interface SwarmTrace {
  retrieval_mode?: string;
  fallback_reason?: string;
  query_plan?: Record<string, unknown>;
  gap_swarms?: GapSwarmRun[];
}

export interface QueryElementReference {
  node_id?: string;
  concept_id?: string;
  entity_id?: string;
  lemma?: string;
  surface?: string;
  lemma_hypotheses?: Array<Record<string, unknown>>;
  morphology?: Record<string, unknown>;
  relation_attachment?: string | null;
  origin?: string;
  confidence?: number;
}

export interface HybridGraphEvidence {
  evidence_id: string;
  source_type: string;
  source_id: string;
  supports?: string[];
  strength: number;
  retrieval_path?: string[];
  independent_source_key?: string;
}

export interface HybridSpatialSupport {
  support_id: string;
  cloud_id: string;
  region_id?: string | null;
  score: number;
  retrieval_path?: string[];
}

export interface HybridFieldRegion {
  center?: number[];
  radius?: number;
  active_dimensions?: string[];
  field_revision?: number;
  region_id?: string;
}

export interface HybridWorkspaceElement {
  element_id: string;
  element_type: string;
  payload?: Record<string, unknown>;
  activation: number;
  workspace_functions?: string[];
  evidence_ids?: string[];
  conflict_ids?: string[];
}

export interface HybridCandidate {
  candidate_id: string;
  gap_id: string;
  element_id: string;
  configuration_id?: string | null;
  event_id?: string | null;
  surface?: string | null;
  lemma?: string | null;
  activation: number;
  score: number;
  status: string;
  evidence_ids?: string[];
  graph_evidence_ids: string[];
  spatial_support_ids: string[];
  independent_source_keys: string[];
  field_fit: number;
  evidential_score?: number;
  constraint_violations?: string[];
}

export interface HybridWorkspace {
  workspace_id: string;
  status: string;
  anchors: HybridWorkspaceElement[];
  active_context: HybridWorkspaceElement[];
  entities: HybridWorkspaceElement[];
  events: HybridWorkspaceElement[];
  scenes: HybridWorkspaceElement[];
  active_clouds: HybridWorkspaceElement[];
  field_region: HybridFieldRegion;
  local_gradients: Array<Record<string, unknown>>;
  graph_evidence: HybridGraphEvidence[];
  spatial_support: HybridSpatialSupport[];
  gaps: Array<{ gap_id: string; surface_projection: string; status: string }>;
  candidates: HybridCandidate[];
  hypotheses: Array<{
    hypothesis_id: string;
    fills: Record<string, string>;
    score: number;
    status: string;
  }>;
  conflicts: Array<{ conflict_id: string; reason: string; severity: number }>;
  evidence: Array<{ evidence_id: string; source_id: string; strength: number }>;
  resonance_state: { iteration: number; stability: number; leader_id: string | null };
  budget: Record<string, number>;
  evictions: Array<Record<string, unknown>>;
  configurations?: Array<Record<string, unknown>>;
  temporal_scope?: Record<string, unknown> | null;
}

export interface HybridPipelineResult {
  query_frame: {
    query_id: string;
    query_type: string;
    known_elements: QueryElementReference[];
    gaps: HybridWorkspace['gaps'];
    exclusions: string[];
    unresolved_context: boolean;
  };
  query_field_projection: {
    anchor_clouds: Array<Record<string, unknown>>;
    positive_gradients: Array<Record<string, unknown>>;
    negative_gradients: Array<Record<string, unknown>>;
    relation_projections: Array<Record<string, unknown>>;
    active_dimensions: string[];
    field_region: HybridFieldRegion;
    [key: string]: unknown;
  };
  retrieval: { hits: Array<{ hit_id: string; element_id: string; element_type: string; match_score: number }>; field_hit_count: number; graph_hit_count: number };
  activation: { activations: Record<string, number>; visited: number; steps: number };
  workspace: HybridWorkspace;
  configurations?: Array<Record<string, unknown>>;
  candidates?: HybridCandidate[];
  hypotheses?: Array<Record<string, unknown>>;
  resonance: { status: string; iterations: number; stable: boolean };
  bees: { decision: { dispatch: boolean; reasons: string[]; task_types: string[]; bee_count: number }; tasks: Array<Record<string, unknown>>; results: Array<Record<string, unknown>> };
  answer_structure: {
    status: string;
    confidence: number;
    filled_gaps: Record<string, string>;
    uncertainties: string[];
    epistemic_mode: string;
    graph_support: number;
    field_support: number;
    independent_source_count: number;
    graph_evidence: string[];
    spatial_support: string[];
    provenance: Record<string, unknown>;
  };
  answer_text: string;
  debug_payload_version?: string;
  trace: { stages: Array<{ stage: string; duration_ms: number; result: string; count?: number }> };
}

export interface BindingConfiguration {
  configuration_id: string;
  event_id: string;
  bindings_by_gap: Record<string, CandidateBinding>;
  all_required_gaps_bound: boolean;
  distinct_node_count: number;
  configuration_score: number;
  graph_validation: GraphAnswer['validation'];
  status: 'SELECTED' | 'REJECTED';
}

export interface HiveSummary {
  id: string;
  conversation_id: string;
  max_cells: number;
  status: string;
}

export interface HiveState {
  hive: HiveSummary;
  query_graph: QueryGraph | null;
  selected_bindings: CandidateBinding[];
  binding_configuration?: BindingConfiguration | null;
  swarm?: SwarmTrace;
  candidate_bindings: CandidateBinding[];
  rejected_events: Array<Record<string, unknown>>;
  answer: GraphAnswer | null;
  trace: Record<string, unknown>;
  hybrid?: HybridPipelineResult;
  turn_index: number;
}

export interface HiveQueryResponse extends Omit<HiveState, 'turn_index'> {
  message_id: string;
  resolved_mode: QueryMode;
  retrieval_scope: RetrievalScope;
}

export interface ChatMessage {
  id: string;
  role: 'user' | 'assistant';
  text: string;
  createdAt: string;
  status?: AnswerStatus | 'PENDING' | 'ERROR';
  queryGraphId?: string;
  answer?: GraphAnswer | null;
  queryGraph?: QueryGraph | null;
}

export interface SlotHypothesis {
  local_slot_id: string;
  compatibility: number;
  evidence: string[];
}

export interface ParticipantNode {
  participant_id: string;
  node_type: 'ENTITY_REFERENCE';
  mention: MentionNode;
  observation_signature: Record<string, number>;
  slot_hypotheses: SlotHypothesis[];
  confidence: number;
}

export interface EventNode {
  event_id: string;
  node_type: 'EVENT';
  predicate: PredicateNode;
  participants: ParticipantNode[];
  properties: Array<Record<string, unknown>>;
  construction_id: string | null;
  polarity: string;
  actuality: string;
  confidence: number;
  raw_text: string;
  versions: ModelVersions;
}

export interface LocalSlot {
  local_slot_id: string;
  predicate_concept_id: string;
  centroid_signature: Record<string, number>;
  support_count: number;
  contradiction_count: number;
  domain_diversity: number;
  confidence: number;
  status: string;
  display_label: string | null;
  display_label_is_non_computational: boolean;
}

export interface SlotSet {
  slot_set_id: string;
  predicate_concept_id: string;
  local_slot_ids: string[];
  support_count: number;
  confidence: number;
  status: string;
}

export interface SlotPrototype {
  prototype_id: string;
  member_slot_ids: string[];
  centroid_signature: Record<string, number>;
  support_count: number;
  domain_diversity: number;
  confidence: number;
  display_label: string | null;
  display_label_is_non_computational: boolean;
}

export interface LanguageAnalysis {
  tokens?: Array<Record<string, unknown>>;
  entity_mentions?: Array<Record<string, unknown>>;
  dialogue_acts?: Array<Record<string, unknown>>;
  interpretation_status?: string;
  [key: string]: unknown;
}

export interface TrainingResponse {
  source_id: string;
  created: boolean;
  status: 'STAGED' | 'CONFIRMED' | 'QUARANTINED' | 'RETRACTED';
  events: EventNode[];
  local_slots?: LocalSlot[];
  slot_sets?: SlotSet[];
  slot_prototypes?: SlotPrototype[];
  language_analysis: LanguageAnalysis;
}
