import { describe, expect, it } from 'vitest';
import { parseStrengthVector } from './format';

describe('parseStrengthVector', () => {
  it('parses comma and semicolon separated values', () => {
    expect(parseStrengthVector('3, 8;2')).toEqual([3, 8, 2]);
  });

  it('drops invalid and negative values', () => {
    expect(parseStrengthVector('3, bad, -5')).toEqual([3, 0]);
  });
});
