export function compactJson(value: unknown): string {
  return JSON.stringify(value, null, 2);
}

export function formatTime(value?: number | null): string {
  if (!value) return '';
  return new Date(value * 1000).toLocaleString();
}

export function parseStrengthVector(value: string): number[] {
  return value
    .split(/[;,]/)
    .map((part) => part.trim())
    .filter(Boolean)
    .map((part) => Math.max(Number.parseInt(part, 10), 0))
    .filter((part) => Number.isFinite(part));
}
