/** Select the text appropriate for a normal dialogue bubble. */

import type { GraphAnswer, QueryGraph } from './types';

function firstText(...values: Array<string | null | undefined>): string {
  return values.find((value): value is string => Boolean(value?.trim())) || 'Ответ не удалось сформировать.';
}

export function getChatAnswerText(
  answer: GraphAnswer | null | undefined,
  queryGraph: QueryGraph | null | undefined,
): string {
  if (!answer) return 'Ответ не удалось сформировать.';
  if (answer.status !== 'RESOLVED') {
    return firstText(answer.surface, answer.short_answer, answer.full_answer);
  }

  const pattern = queryGraph?.event_pattern;
  const gapCount = Array.isArray(pattern?.target_gaps)
    ? pattern.target_gaps.length
    : pattern?.target_gap ? 1 : 0;

  return gapCount === 1
    ? firstText(answer.surface, answer.short_answer, answer.full_answer)
    : firstText(answer.full_answer, answer.surface, answer.short_answer);
}
