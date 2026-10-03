/**
 * Risk ruler: the model's probability on a 0–100% measurement scale, with the
 * decision threshold and the bootstrap uncertainty band.
 *
 * The SVG is drawn at the container's real pixel width (measured with
 * ResizeObserver) so text stays readable on phones and large screens alike.
 * Optional `markers` draws extra points on the same scale (e.g. other models).
 */
import { useEffect, useRef, useState } from 'react';

function useWidth() {
  const ref = useRef(null);
  const [width, setWidth] = useState(720);
  useEffect(() => {
    if (!ref.current || typeof ResizeObserver === 'undefined') return undefined;
    const ro = new ResizeObserver(([entry]) => setWidth(Math.max(240, entry.contentRect.width)));
    ro.observe(ref.current);
    return () => ro.disconnect();
  }, []);
  return [ref, width];
}

export default function RiskRuler({ probability, threshold, lower, upper, markers = null, compact = false, label }) {
  const [ref, W] = useWidth();
  const PAD = 16;
  const x = (p) => PAD + Math.max(0, Math.min(1, p)) * (W - 2 * PAD);
  const narrow = W < 520;
  const axisY = 44;
  const H = markers ? axisY + 100 : axisY + 40;
  const high = probability >= threshold;
  const color = high ? 'var(--high)' : 'var(--low)';
  const step = narrow ? 0.25 : 0.1;
  const ticks = Array.from({ length: Math.round(1 / step) + 1 }, (_, i) => +(i * step).toFixed(2));
  const minor = Array.from({ length: 21 }, (_, i) => i / 20).filter((t) => !ticks.includes(+t.toFixed(2)));
  const desc = `Model probability ${(probability * 100).toFixed(1)}%, threshold ${(threshold * 100).toFixed(1)}%` +
    (lower !== undefined ? `, bootstrap range ${(lower * 100).toFixed(1)}% to ${(upper * 100).toFixed(1)}%` : '');
  const thrAnchor = threshold > 0.85 ? 'end' : threshold < 0.15 ? 'start' : 'middle';

  return (
    <figure className="ruler" style={{ margin: 0 }} ref={ref}>
      <svg width={W} height={H} viewBox={`0 0 ${W} ${H}`} role="img" aria-label={label ? `${label}: ${desc}` : desc}>
        <rect x={x(0)} y={axisY - 10} width={x(threshold) - x(0)} height={20} fill="var(--low-tint)" />
        <rect x={x(threshold)} y={axisY - 10} width={x(1) - x(threshold)} height={20} fill="var(--high-tint)" />

        {lower !== undefined && upper !== undefined && (
          <rect x={x(lower)} y={axisY - 5} width={Math.max(2, x(upper) - x(lower))} height={10}
                fill="var(--band)" opacity="0.8" rx="2" />
        )}

        <line x1={x(0)} x2={x(1)} y1={axisY + 10} y2={axisY + 10} stroke="var(--ink)" strokeWidth="1.5" />
        {minor.map((t) => (
          <line key={`m${t}`} x1={x(t)} x2={x(t)} y1={axisY + 10} y2={axisY + 14} stroke="var(--rule-strong)" />
        ))}
        {ticks.map((t) => (
          <g key={t}>
            <line x1={x(t)} x2={x(t)} y1={axisY + 10} y2={axisY + 17} stroke="var(--ink)" />
            <text x={x(t)} y={axisY + 30} textAnchor="middle" fontSize="12" fill="var(--muted)">
              {Math.round(t * 100)}%
            </text>
          </g>
        ))}

        <line x1={x(threshold)} x2={x(threshold)} y1={axisY - 20} y2={axisY + 12}
              stroke="var(--ink)" strokeWidth="1.5" strokeDasharray="4 3" />
        <text x={x(threshold)} y={axisY - 25} textAnchor={thrAnchor} fontSize="12" fill="var(--ink)">
          threshold {(threshold * 100).toFixed(0)}%
        </text>

        <line x1={x(probability)} x2={x(probability)} y1={axisY - 14} y2={axisY + 14} stroke={color} strokeWidth="4" />
        <circle cx={x(probability)} cy={axisY} r="7" fill={color} stroke="#fff" strokeWidth="2.5" />

        {markers && markers.map((m, i) => {
          const y = axisY + 50 + (i % 3) * 15;
          return (
            <g key={m.key}>
              <line x1={x(m.probability)} x2={x(m.probability)} y1={axisY + 18} y2={y - 5} stroke="var(--rule-strong)" />
              <circle cx={x(m.probability)} cy={y} r="5"
                      fill={m.probability >= threshold ? 'var(--high)' : 'var(--low)'} opacity="0.85" />
              <title>{`${m.display_name}: ${(m.probability * 100).toFixed(1)}%`}</title>
            </g>
          );
        })}
      </svg>
      {!compact && (
        <figcaption className="ruler-legend">
          <span><i style={{ background: color }} />Model estimate</span>
          {lower !== undefined && <span><i style={{ background: 'var(--band)' }} />Bootstrap 90% range</span>}
          <span><i style={{ background: 'var(--low-tint)', border: '1px solid var(--rule)' }} />Below threshold</span>
          <span><i style={{ background: 'var(--high-tint)', border: '1px solid var(--rule)' }} />Above threshold</span>
        </figcaption>
      )}
    </figure>
  );
}
