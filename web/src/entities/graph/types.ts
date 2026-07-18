/** Public client contracts for the role-free SuperAI V2.7 graph API. */

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
  question_signature: Record<string, number>;
  compatible_slot_hypotheses: Record<string, number>;
}

export interface QueryGraph {
  query_graph_id: string;
  event_pattern: {
    predicate: PredicateNode | null;
    known_nodes: MentionNode[];
    gap_node: GapNode;
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
  selected_binding?: CandidateBinding;
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

export interface HiveSummary {
  id: string;
  conversation_id: string;
  max_cells: number;
  status: string;
}

export interface HiveState {
  hive: HiveSummary;
  query_graph: QueryGraph | null;
  selected_binding: CandidateBinding | null;
  candidate_bindings: CandidateBinding[];
  rejected_events: Array<Record<string, unknown>>;
  answer: GraphAnswer | null;
  trace: Record<string, unknown>;
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
