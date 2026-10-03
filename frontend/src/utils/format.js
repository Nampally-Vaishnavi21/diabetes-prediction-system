export const pct = (p, digits = 0) =>
  p === null || p === undefined ? '–' : `${(p * 100).toFixed(digits)}%`;

export const num = (x, digits = 3) =>
  x === null || x === undefined || Number.isNaN(x) ? '–' : Number(x).toFixed(digits);

export const signed = (x, digits = 3) =>
  x === null || x === undefined ? '–' : `${x > 0 ? '+' : x < 0 ? '−' : '±'}${Math.abs(x).toFixed(digits)}`;

export const pp = (x, digits = 1) =>
  `${x > 0 ? '+' : x < 0 ? '−' : '±'}${Math.abs(x).toFixed(digits)} pp`;

export const date = (iso) => {
  if (!iso) return '–';
  const d = new Date(iso);
  return Number.isNaN(d.getTime()) ? iso : d.toLocaleString();
};

/** Muted, distinguishable line colours for the 7 compared models. */
export const MODEL_COLORS = ['#14283a', '#2d7a68', '#972a3a', '#1f5fa8', '#b5761b', '#6b4c9a', '#5a6e7c'];
