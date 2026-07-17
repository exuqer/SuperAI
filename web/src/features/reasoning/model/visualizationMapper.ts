import type { QuerySceneV2, QuerySceneCandidateV2 } from '@/entities/hive/types';

export type HiveMode =
  | 'whole'
  | 'parse'
  | 'event'
  | 'scene'
  | 'roles'
  | 'search'
  | 'candidates'
  | 'concepts'
  | 'concept-evidence'
  | 'structure'
  | 'answer'
  | 'resonance'
  | 'dynamics'
  | 'multilevel';

const roleLabel: Record<string, string> = { agent: 'AGENT', action: 'ACTION', predicate: 'ACTION', modal: 'MODAL', object: 'OBJECT', location: 'LOCATION', time: 'TIME', instrument: 'INSTRUMENT', subject: 'SUBJECT' };

export function mapVisualization(source: Record<string, any>) {
  const scene = source.queryScene as QuerySceneV2 | null;
  const slots = (scene?.slots || []).map((slot: any) => ({
    ...slot,
    role: roleLabel[slot.role] || String(slot.role || '').toUpperCase(),
    lemma: slot.status?.toUpperCase?.() === 'RESOLVED' ? (slot.value?.lemma || slot.lemma || slot.surface) : slot.lemma || slot.local_lemma || slot.label || slot.surface || slot.question_word || '—',
    surface: slot.surface || slot.form || slot.label || '',
    status: String(slot.status || 'empty').toUpperCase(),
    candidates: slot.candidates || [],
  }));
  const object = slots.find((slot: any) => slot.role === 'OBJECT');
  const bridge = source.unknownTokenSearches?.[0] || null;
  const candidateSource = (source.queryCandidates || []) as QuerySceneCandidateV2[];
  const candidates = candidateSource.map((candidate: any) => ({
    ...candidate,
    score: Math.round(Number(candidate.scores?.total ?? candidate.score ?? .5) * 100),
    label: candidate.lemma || candidate.surface,
  }));
  const selectedSourceIds = new Set((source.memorySources || []).map((item: any) => String(item.source_scene_id || item.id || '')));
  const sources = (source.memoryScenes || []).map((item: any) => ({
    ...item,
    text: item.source_text || item.sentence_text || item.label || 'Источник памяти',
    score: Math.round(Number(item.scores?.total_score ?? item.score ?? item.total_score ?? 0) * 100),
    selected: selectedSourceIds.has(String(item.cloud_id || String(item.id || '').replace('scene-', ''))),
    anchorValidation: item.anchor_validation || item.scores?.anchor_validation || {},
    roleMatchDetails: item.role_match_details || item.scores?.role_match_details || {},
    conceptSpaceIds: Array.from(new Set(Object.values(item.role_match_details || item.scores?.role_match_details || {}).flatMap((detail: any) => detail?.concept_space_ids || []))),
    semanticApproximation: Object.values(item.role_match_details || item.scores?.role_match_details || {}).some((detail: any) => ['stable_concept', 'related_concept', 'shared_category'].includes(detail?.match_type)),
  }));
  return {
    scene: {
    activeQuery: source.activeQuery?.text || source.queryFrame?.source_text || source.queryScene?.source_query || '',
    slots,
    modal: slots.find((slot: any) => slot.role === 'MODAL') || null,
      bridge: object && bridge ? {
        surface: object.surface || bridge.surface || '',
        lemma: object.lemma || bridge.lemma_hypotheses?.[0]?.lemma || '',
        global: bridge.selected_candidate?.candidate_lexeme || bridge.selected_semantic_candidate?.candidate_lexeme || '',
        confidence: Math.round(Number(bridge.candidate_bridges?.[0]?.confidence ?? bridge.selected_candidate?.scores?.semantic_total ?? 0) * 100),
        sharedBase: bridge.candidate_bridges?.[0]?.shared_base || bridge.selected_semantic_candidate?.shared_base || '',
      } : null,
    },
    search: {
      missions: (bridge?.bee_missions || []).map((mission: any) => [
        mission.bee_type || mission.id,
        mission.source_level || mission.bee_type || 'Поиск',
        mission.result || mission.status || '—',
      ]),
      candidates,
      sources,
      counts: { found: sources.length, validated: sources.filter((item: any) => item.anchorValidation?.status === 'PASSED').length, candidates: candidates.length },
      history: source.vibrationHistory || [],
    },
    analysis: {
      parse: {
        source_text: source.queryFrame?.source_text || '',
        tokens: source.queryFrame?.tokens || [],
        slots: source.queryScene?.slots || [],
      },
      events: sources.map((item: any) => ({
        scene_id: item.id,
        source_text: item.text,
        event: item.event || null,
        entity_mentions: item.entity_mentions || [],
      })),
      roleHypotheses: {
        requested_role: source.queryFrame?.requested_role || null,
        requested_slot: source.queryFrame?.requested_slot || null,
        query: source.queryFrame?.requested_role_hypotheses || [],
        events: sources.flatMap((item: any) =>
          (item.event?.participants || []).map((participant: any) => ({
            scene_id: item.id,
            entity: participant.surface,
            grammatical_slot: participant.grammatical_slot,
            selected_role: participant.role,
            hypotheses: participant.role_hypotheses || [],
          }))
        ),
      },
      candidates: {
        accepted: candidateSource,
        rejected: sources
          .filter((item: any) => item.candidate_validation || item.candidate_status === 'rejected')
          .map((item: any) => item.candidate_validation || {
            scene_id: item.id,
            reason_code: item.decision_reason || 'CONSTRAINT_FAILED',
          }),
      },
      concepts: {
        query: source.queryFrame?.conceptual_query_frame || null,
        projections: sources
          .filter((item: any) => item.scene_concept_projection || item.event)
          .map((item: any) => ({
            scene_id: item.id,
            projection: item.scene_concept_projection || null,
            construction_id: item.event?.construction_id || null,
          })),
      },
      conceptEvidence: sources.flatMap((item: any) =>
        item.concept_evidence || item.event?.concept_evidence || []
      ),
    },
    answerBuild: {
      shortPlan: source.sentencePlan || null,
      fullPlan: source.fullSentencePlan || null,
      generationCandidates: source.generationCandidates || [],
      morphologyTrace: source.morphologyTrace || [],
      reverseValidation: source.reverseValidation || { status: source.queryAnswer?.status === 'BUILD_FAILED' ? 'FAILED' : source.queryAnswer?.status === 'RESOLVED' ? 'PASSED' : 'WAITING', score: source.queryAnswer?.status === 'RESOLVED' ? 1 : 0 },
      answer: source.queryAnswer?.surface_answer || '',
      fullAnswer: source.queryAnswer?.full_surface_answer || '',
    },
    resonance: {
      ...(source.localResonance || { status: 'WAITING', probe_text: '', matched_lexeme: null, matched_form: null }),
      probe: source.resonanceProbes?.find((item: any) => item.id === source.localResonance?.latest_probe_id) || source.resonanceProbes?.at?.(-1) || null,
    },
    hiveStructure: source.hiveStructure || { placements: { working_cells: 0, memory_sources: sources.length, total: 0 }, working_items: [], sources, selected_structure_target: null },
  };
}
