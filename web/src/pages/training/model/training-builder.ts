import { parseStrengthVector } from '@/shared/lib/format';

export const TRAINING_TOP_DOMAINS = [
  { key: 'dialogue', label: 'Общение', uri: '/m/top/dialogue' },
  { key: 'object', label: 'Предмет', uri: '/m/top/object' },
  { key: 'action', label: 'Действие', uri: '/m/top/action' },
  { key: 'person', label: 'Человек', uri: '/m/top/person' },
  { key: 'place', label: 'Место', uri: '/m/top/place' },
  { key: 'emotion', label: 'Эмоция', uri: '/m/top/emotion' },
  { key: 'language', label: 'Язык', uri: '/m/top/language' },
  { key: 'number', label: 'Число', uri: '/m/top/number' },
  { key: 'nature', label: 'Природа', uri: '/m/top/nature' },
  { key: 'mind', label: 'Мышление', uri: '/m/top/mind' },
  { key: 'perception', label: 'Восприятие', uri: '/m/top/perception' },
  { key: 'body', label: 'Тело', uri: '/m/top/body' },
] as const;

export type TrainingTopDomainKey = (typeof TRAINING_TOP_DOMAINS)[number]['key'];

export type TrainingLayerDraft = {
  level: number;
  label: string;
  builtinTopDomain?: TrainingTopDomainKey;
};

export type TrainingExampleDraft = {
  question: string;
  expectedAnswer: string;
  lang: string;
  strengthVector: number[];
  layers: TrainingLayerDraft[];
};

export function createDefaultTrainingDraft(): TrainingExampleDraft {
  return {
    question: 'как дела?',
    expectedAnswer: 'Нормально, спасибо. А у тебя?',
    lang: 'ru',
    strengthVector: [3, 8, 8],
    layers: [
      { level: 0, label: 'Общение', builtinTopDomain: 'dialogue' },
      { level: 1, label: 'Вопрос' },
      { level: 2, label: 'дела' },
    ],
  };
}

export function parseTrainingStrengthVector(value: string, fallback: number[] = [3, 8, 8]): number[] {
  return normalizeStrengthVector(parseStrengthVector(value), fallback);
}

export function buildTrainingExampleJsonl(draft: TrainingExampleDraft): string {
  return JSON.stringify(buildTrainingExamplePayload(draft));
}

export function buildTrainingExamplePayload(draft: TrainingExampleDraft): Record<string, unknown> {
  const layers = draft.layers.map((layer) => resolveLayerTarget(draft.lang, layer));
  const strengthVector = normalizeStrengthVector(draft.strengthVector);
  const layerTargets: Record<string, string[]> = {};
  const targetConcepts: string[] = [];
  const conceptLabels: Record<string, string> = {};

  layers.forEach((layer, index) => {
    layerTargets[String(index)] = [layer.uri];
    if (!targetConcepts.includes(layer.uri)) {
      targetConcepts.push(layer.uri);
    }
    conceptLabels[layer.uri] = layer.label;
  });

  return {
    stimulus: normalizeText(draft.question),
    lang: draft.lang,
    strength_vector: strengthVector,
    layer_targets: layerTargets,
    target_concepts: targetConcepts,
    concept_labels: conceptLabels,
    accepted_answer: normalizeText(draft.expectedAnswer),
    metadata: {
      source: 'web_training_form',
      kind: 'qa_with_layers',
    },
  };
}

export function resolveLayerTarget(
  lang: string,
  layer: TrainingLayerDraft,
): { level: number; label: string; uri: string } {
  if (layer.level === 0) {
    const domain = TRAINING_TOP_DOMAINS.find((item) => item.key === layer.builtinTopDomain) ?? TRAINING_TOP_DOMAINS[0];
    return {
      level: 0,
      label: domain.label,
      uri: domain.uri,
    };
  }
  const label = normalizeText(layer.label) || `Плоскость ${layer.level}`;
  return {
    level: layer.level,
    label,
    uri: `/m/user/${slugSegment(lang)}/${slugSegment(label) || `layer-${layer.level}`}`,
  };
}

function normalizeText(value: string): string {
  return value.replace(/\s+/g, ' ').trim();
}

function normalizeStrengthVector(
  value: readonly number[],
  fallback: number[] = [3, 8, 8],
): number[] {
  const source = value.length ? value : fallback;
  return source.map((item, index) => {
    const numeric = Number(item);
    const fallbackValue = Number.isFinite(Number(fallback[index])) ? Math.trunc(Number(fallback[index])) : 0;
    return Number.isFinite(numeric) ? Math.max(Math.trunc(numeric), 0) : fallbackValue;
  });
}

function slugSegment(value: string): string {
  return normalizeText(value)
    .toLowerCase()
    .normalize('NFKC')
    .replace(/['"`’]/g, '')
    .replace(/[^\p{Letter}\p{Number}]+/gu, '_')
    .replace(/^_+|_+$/g, '')
    .replace(/_+/g, '_');
}
