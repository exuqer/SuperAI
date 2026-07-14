/** Geometry utilities for graph visualizations. */

export interface Point {
  x: number;
  y: number;
}

export interface Rect {
  x: number;
  y: number;
  width: number;
  height: number;
}

export function distance(a: Point, b: Point): number {
  return Math.hypot(a.x - b.x, a.y - b.y);
}

export function clamp(value: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, value));
}

export function lerp(a: number, b: number, t: number): number {
  return a + (b - a) * clamp(t, 0, 1);
}

export function polarToCartesian(centerX: number, centerY: number, radius: number, angle: number): Point {
  return {
    x: centerX + radius * Math.cos(angle),
    y: centerY + radius * Math.sin(angle),
  };
}

export function cartesianToPolar(centerX: number, centerY: number, x: number, y: number): { radius: number; angle: number } {
  const dx = x - centerX;
  const dy = y - centerY;
  return {
    radius: Math.hypot(dx, dy),
    angle: Math.atan2(dy, dx),
  };
}

export function getAngleBetween(a: Point, b: Point): number {
  return Math.atan2(b.y - a.y, b.x - a.x);
}

export function rotatePoint(point: Point, center: Point, angle: number): Point {
  const cos = Math.cos(angle);
  const sin = Math.sin(angle);
  const dx = point.x - center.x;
  const dy = point.y - center.y;
  return {
    x: center.x + dx * cos - dy * sin,
    y: center.y + dx * sin + dy * cos,
  };
}

export function generateStablePosition(namespace: string, index: number, radius: number = 100): Point {
  // Deterministic pseudo-random position based on namespace and index
  const seed = hashCode(`${namespace}:${index}`);
  const angle = (seed * 2.399963) % (2 * Math.PI);
  return polarToCartesian(500, 350, radius, angle);
}

function hashCode(str: string): number {
  let hash = 0;
  for (let i = 0; i < str.length; i++) {
    const char = str.charCodeAt(i);
    hash = ((hash << 5) - hash) + char;
    hash = hash & hash; // Convert to 32bit integer
  }
  return Math.abs(hash);
}

export function getBounds(points: Point[]): Rect | null {
  if (points.length === 0) return null;
  let minX = points[0].x, maxX = points[0].x;
  let minY = points[0].y, maxY = points[0].y;
  for (const p of points) {
    minX = Math.min(minX, p.x);
    maxX = Math.max(maxX, p.x);
    minY = Math.min(minY, p.y);
    maxY = Math.max(maxY, p.y);
  }
  return { x: minX, y: minY, width: maxX - minX, height: maxY - minY };
}

export function fitToView(points: Point[], viewWidth: number, viewHeight: number, padding: number = 50): { scale: number; offset: Point } {
  const bounds = getBounds(points);
  if (!bounds) return { scale: 1, offset: { x: viewWidth / 2, y: viewHeight / 2 } };

  const scaleX = (viewWidth - 2 * padding) / Math.max(1, bounds.width);
  const scaleY = (viewHeight - 2 * padding) / Math.max(1, bounds.height);
  const scale = Math.min(scaleX, scaleY, 5); // Max 5x zoom

  const centerX = bounds.x + bounds.width / 2;
  const centerY = bounds.y + bounds.height / 2;

  return {
    scale,
    offset: {
      x: viewWidth / 2 - centerX * scale,
      y: viewHeight / 2 - centerY * scale,
    },
  };
}