import { describe, expect, it } from 'vitest';
import { buildTrainingExampleJsonl, createDefaultTrainingDraft } from './training-builder';

describe('training builder', () => {
  it('builds canonical jsonl for the default layered qa draft', () => {
    const draft = createDefaultTrainingDraft();

    expect(buildTrainingExampleJsonl(draft)).toBe(
      '{"stimulus":"как дела?","lang":"ru","strength_vector":[3,8,8],"layer_targets":{"0":["/m/top/dialogue"],"1":["/m/user/ru/вопрос"],"2":["/m/user/ru/дела"]},"target_concepts":["/m/top/dialogue","/m/user/ru/вопрос","/m/user/ru/дела"],"concept_labels":{"/m/top/dialogue":"Общение","/m/user/ru/вопрос":"Вопрос","/m/user/ru/дела":"дела"},"accepted_answer":"Нормально, спасибо. А у тебя?","metadata":{"source":"web_training_form","kind":"qa_with_layers"}}',
    );
  });
});
