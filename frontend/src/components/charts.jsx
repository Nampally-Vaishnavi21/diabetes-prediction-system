/**
 * Chart components. Every chart is drawn from arrays returned by the API —
 * there is no sample or generated data in the frontend.
 */
import {
  Area, Bar, BarChart, CartesianGrid, Cell, ComposedChart, Legend, Line, LineChart,
  ReferenceLine, ResponsiveContainer, Scatter, ScatterChart, Tooltip, XAxis, YAxis, ZAxis,
} from 'recharts';
import { useEffect, useState } from 'react';
import { MODEL_COLORS, num, pct, signed } from '../utils/format';

/** True on phone-width screens: charts then use a narrower label column. */
function useNarrow(query = '(max-width: 600px)') {
  const get = () => typeof window !== 'undefined' && window.matchMedia?.(query).matches;
  const [narrow, setNarrow] = useState(get);
  useEffect(() => {
    if (!window.matchMedia) return undefined;
    const mq = window.matchMedia(query);
    const on = () => setNarrow(mq.matches);
    mq.addEventListener('change', on);
    return () => mq.removeEventListener('change', on);
  }, [query]);
  return narrow;
}

const AXIS = { fontSize: 12, fill: '#5a6e7c' };
const GRID = <CartesianGrid stroke="#e3eaec" strokeDasharray="3 3" />;

/* ---------- SHAP: local contributions (diverging horizontal bars) ---------- */
export function ShapBarChart({ contributions, height }) {
  const labelWidth = useNarrow() ? 112 : 190;
  const data = contributions.map((c) => ({
    name: c.label + (c.imputed ? ' (imputed)' : ''),
    value: c.shap_value,
    raw: c.value,
    unit: c.unit,
  }));
  return (
    <div className="chart-box" style={{ height: height || 44 * data.length + 60 }}>
      <ResponsiveContainer>
        <BarChart data={data} layout="vertical" margin={{ top: 8, right: 24, left: 8, bottom: 8 }}>
          {GRID}
          <XAxis type="number" tick={AXIS} tickFormatter={(v) => signed(v, 2)}
                 label={{ value: 'Contribution to probability', position: 'insideBottom', offset: -4, ...AXIS }} />
          <YAxis type="category" dataKey="name" width={labelWidth} tick={AXIS} />
          <ReferenceLine x={0} stroke="#14283a" />
          <Tooltip
            formatter={(v) => [signed(v, 4), 'SHAP contribution']}
            labelFormatter={(l, p) => {
              const d = p?.[0]?.payload;
              return d ? `${l}: ${d.raw ?? 'blank'} ${d.raw !== null && d.raw !== undefined ? d.unit : ''}` : l;
            }}
          />
          <Bar dataKey="value" isAnimationActive={false} barSize={18}>
            {data.map((d) => <Cell key={d.name} fill={d.value >= 0 ? '#972a3a' : '#2d7a68'} />)}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

export function ContributionTable({ explanation }) {
  return (
    <div className="table-wrap">
      <table>
        <caption className="visually-hidden">SHAP contributions per feature</caption>
        <thead>
          <tr><th>Feature</th><th>Patient value</th><th>Contribution</th><th>Effect on estimate</th></tr>
        </thead>
        <tbody>
          {explanation.contributions.map((c) => (
            <tr key={c.feature}>
              <td>{c.label}</td>
              <td>{c.imputed ? <span className="tag">blank, imputed</span> : `${c.value} ${c.unit}`}</td>
              <td><span className={c.shap_value >= 0 ? 'pos' : 'neg'}>{signed(c.shap_value, 4)}</span></td>
              <td>{c.direction === 'increases' ? 'Raises' : c.direction === 'decreases' ? 'Lowers' : 'No effect'}</td>
            </tr>
          ))}
          <tr>
            <td>Base value (average model output)</td><td></td><td>{num(explanation.base_value, 4)}</td><td></td>
          </tr>
          <tr>
            <td><b>Base + contributions = model probability</b></td><td></td>
            <td><b>{num(explanation.sum_check, 4)}</b></td><td>{pct(explanation.prediction, 1)}</td>
          </tr>
        </tbody>
      </table>
    </div>
  );
}

/* ---------- SHAP: global importance ---------- */
export function ImportanceChart({ importance }) {
  const labelWidth = useNarrow() ? 112 : 190;
  const data = importance.map((d) => ({ name: d.label, value: d.mean_abs_shap }));
  return (
    <div className="chart-box" style={{ height: 40 * data.length + 60 }}>
      <ResponsiveContainer>
        <BarChart data={data} layout="vertical" margin={{ top: 8, right: 24, left: 8, bottom: 8 }}>
          {GRID}
          <XAxis type="number" tick={AXIS} tickFormatter={(v) => v.toFixed(2)}
                 label={{ value: 'Mean |SHAP| (probability)', position: 'insideBottom', offset: -4, ...AXIS }} />
          <YAxis type="category" dataKey="name" width={labelWidth} tick={AXIS} />
          <Tooltip formatter={(v) => [v.toFixed(4), 'Mean |SHAP|']} />
          <Bar dataKey="value" fill="#14283a" isAnimationActive={false} barSize={16} />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

/* ---------- SHAP: beeswarm (summary) ---------- */
function lerpColor(t) {
  // low feature value -> teal, high -> oxblood, missing -> grey
  if (t === null || t === undefined) return '#a9bac0';
  const a = [45, 122, 104];
  const b = [151, 42, 58];
  const c = a.map((v, i) => Math.round(v + (b[i] - v) * t));
  return `rgb(${c.join(',')})`;
}

export function BeeswarmChart({ beeswarm }) {
  const labelWidth = useNarrow() ? 112 : 190;
  const labels = beeswarm.map((f) => f.label);
  const points = [];
  beeswarm.forEach((f, row) => {
    // deterministic vertical jitter: spread points sharing similar SHAP values
    const sorted = [...f.points].sort((p, q) => p.shap - q.shap);
    sorted.forEach((p, i) => {
      const jitter = ((i * 7919) % 100) / 100 - 0.5;
      points.push({ x: p.shap, y: row + jitter * 0.6, value: p.value, norm: p.norm, feature: f.label });
    });
  });
  return (
    <div className="chart-box tall" style={{ height: 46 * labels.length + 70 }}>
      <ResponsiveContainer>
        <ScatterChart margin={{ top: 8, right: 24, left: 8, bottom: 16 }}>
          {GRID}
          <XAxis type="number" dataKey="x" tick={AXIS} tickFormatter={(v) => signed(v, 2)}
                 label={{ value: 'SHAP contribution (probability)', position: 'insideBottom', offset: -8, ...AXIS }} />
          <YAxis type="number" dataKey="y" domain={[-0.6, labels.length - 0.4]} reversed
                 ticks={labels.map((_, i) => i)} tickFormatter={(i) => labels[i] ?? ''}
                 width={labelWidth} tick={AXIS} interval={0} />
          <ZAxis range={[22, 22]} />
          <ReferenceLine x={0} stroke="#14283a" />
          <Tooltip
            cursor={false}
            content={({ payload }) => {
              const d = payload?.[0]?.payload;
              if (!d) return null;
              return (
                <div className="panel" style={{ padding: 8, fontSize: 12 }}>
                  {d.feature}: {d.value ?? 'missing'}<br />SHAP {signed(d.x, 4)}
                </div>
              );
            }}
          />
          <Scatter data={points} isAnimationActive={false}>
            {points.map((p, i) => <Cell key={i} fill={lerpColor(p.norm)} fillOpacity={0.8} />)}
          </Scatter>
        </ScatterChart>
      </ResponsiveContainer>
    </div>
  );
}

/* ---------- ROC / PR curves for several models ---------- */
export function MultiCurveChart({ series, xKey, yKey, xLabel, yLabel, diagonal = false, baseline = null }) {
  return (
    <div className="chart-box tall">
      <ResponsiveContainer>
        <LineChart margin={{ top: 8, right: 16, left: 0, bottom: 16 }}>
          {GRID}
          <XAxis type="number" dataKey={xKey} domain={[0, 1]} tick={AXIS} allowDuplicatedCategory={false}
                 label={{ value: xLabel, position: 'insideBottom', offset: -8, ...AXIS }} />
          <YAxis type="number" domain={[0, 1]} tick={AXIS}
                 label={{ value: yLabel, angle: -90, position: 'insideLeft', offset: 12, ...AXIS }} />
          <Tooltip formatter={(v) => num(v, 3)} labelFormatter={(l) => `${xLabel}: ${num(l, 3)}`} />
          <Legend wrapperStyle={{ fontSize: 12 }} />
          {diagonal && (
            <Line data={[{ [xKey]: 0, [yKey]: 0 }, { [xKey]: 1, [yKey]: 1 }]} dataKey={yKey}
                  name="Chance" stroke="#a9bac0" strokeDasharray="4 4" dot={false} isAnimationActive={false} />
          )}
          {baseline !== null && <ReferenceLine y={baseline} stroke="#a9bac0" strokeDasharray="4 4"
                                               label={{ value: `Baseline ${baseline.toFixed(2)}`, ...AXIS, position: 'insideTopRight' }} />}
          {series.map((s, i) => (
            <Line key={s.name} data={s.points} dataKey={yKey} name={s.name}
                  stroke={s.color || MODEL_COLORS[i % MODEL_COLORS.length]}
                  strokeWidth={s.highlight ? 3 : 1.5} dot={false} isAnimationActive={false}
                  type={yKey === 'precision' ? 'stepAfter' : 'linear'} />
          ))}
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}

/* ---------- calibration (reliability) curve ---------- */
export function CalibrationChart({ series }) {
  return (
    <div className="chart-box tall">
      <ResponsiveContainer>
        <LineChart margin={{ top: 8, right: 16, left: 0, bottom: 16 }}>
          {GRID}
          <XAxis type="number" dataKey="mean_predicted" domain={[0, 1]} tick={AXIS}
                 label={{ value: 'Mean predicted probability', position: 'insideBottom', offset: -8, ...AXIS }} />
          <YAxis type="number" domain={[0, 1]} tick={AXIS}
                 label={{ value: 'Observed fraction diabetic', angle: -90, position: 'insideLeft', offset: 12, ...AXIS }} />
          <Tooltip formatter={(v) => num(v, 3)} labelFormatter={(l) => `Predicted: ${num(l, 3)}`} />
          <Legend wrapperStyle={{ fontSize: 12 }} />
          <Line data={[{ mean_predicted: 0, fraction_positive: 0 }, { mean_predicted: 1, fraction_positive: 1 }]}
                dataKey="fraction_positive" name="Perfect calibration" stroke="#a9bac0"
                strokeDasharray="4 4" dot={false} isAnimationActive={false} />
          {series.map((s, i) => (
            <Line key={s.name} data={s.points} dataKey="fraction_positive" name={s.name}
                  stroke={[ '#14283a', '#b5761b'][i % 2]} strokeWidth={2} isAnimationActive={false} />
          ))}
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}

/* ---------- confusion matrix ---------- */
export function ConfusionMatrix({ cm }) {
  const max = Math.max(cm.tn, cm.fp, cm.fn, cm.tp);
  const cell = (v, good) => ({
    background: good ? `rgba(45,122,104,${0.12 + 0.6 * v / max})` : `rgba(151,42,58,${0.08 + 0.5 * v / max})`,
  });
  return (
    <div className="cm" role="table" aria-label="Confusion matrix">
      <div className="h" />
      <div className="h">Predicted lower risk (0)</div>
      <div className="h">Predicted higher risk (1)</div>
      <div className="h" style={{ textAlign: 'right' }}>Actual 0</div>
      <div style={cell(cm.tn, true)}><span className="c">{cm.tn}</span><br /><small>true negatives</small></div>
      <div style={cell(cm.fp, false)}><span className="c">{cm.fp}</span><br /><small>false positives</small></div>
      <div className="h" style={{ textAlign: 'right' }}>Actual 1</div>
      <div style={cell(cm.fn, false)}><span className="c">{cm.fn}</span><br /><small>false negatives</small></div>
      <div style={cell(cm.tp, true)}><span className="c">{cm.tp}</span><br /><small>true positives</small></div>
    </div>
  );
}

/* ---------- correlation heatmap ---------- */
export function CorrelationHeatmap({ matrix }) {
  const n = matrix.columns.length;
  const color = (v) => (v >= 0 ? `rgba(151,42,58,${Math.abs(v)})` : `rgba(31,95,168,${Math.abs(v)})`);
  return (
    <div className="heat" style={{ gridTemplateColumns: `auto repeat(${n}, minmax(44px, 1fr))` }}
         role="table" aria-label="Correlation matrix">
      <div className="lab" />
      {matrix.columns.map((c) => <div key={c} className="lab" style={{ textAlign: 'center' }} title={c}>{c.slice(0, 8)}</div>)}
      {matrix.values.map((row, i) => [
        <div key={`l${i}`} className="lab">{matrix.columns[i]}</div>,
        ...row.map((v, j) => (
          <div key={`${i}-${j}`} style={{ background: color(v), color: Math.abs(v) > 0.55 ? '#fff' : '#14283a' }}
               title={`${matrix.columns[i]} × ${matrix.columns[j]}: ${v}`}>
            {v.toFixed(2)}
          </div>
        )),
      ])}
    </div>
  );
}

/* ---------- what-if sensitivity curve ---------- */
export function SensitivityChart({ data }) {
  const rows = data.points.map((p) => ({ ...p, band: [p.lower, p.upper] }));
  return (
    <div className="chart-box tall">
      <ResponsiveContainer>
        <ComposedChart data={rows} margin={{ top: 16, right: 16, left: 0, bottom: 16 }}>
          {GRID}
          <XAxis type="number" dataKey="value" domain={['dataMin', 'dataMax']} tick={AXIS}
                 label={{ value: `${data.label} (${data.unit})`, position: 'insideBottom', offset: -8, ...AXIS }} />
          <YAxis domain={[0, 1]} tick={AXIS} tickFormatter={(v) => pct(v)}
                 label={{ value: 'Model probability', angle: -90, position: 'insideLeft', offset: 12, ...AXIS }} />
          <Tooltip
            formatter={(v, name) => (Array.isArray(v) ? [`${pct(v[0], 1)} – ${pct(v[1], 1)}`, name] : [pct(v, 1), name])}
            labelFormatter={(l) => `${data.label}: ${num(l, 2)} ${data.unit}`}
          />
          <Area dataKey="band" name="Bootstrap 90% range" stroke="none" fill="#9fb4bf" fillOpacity={0.45}
                isAnimationActive={false} />
          <Line dataKey="probability" name="Model probability" stroke="#14283a" strokeWidth={2.5}
                dot={false} isAnimationActive={false} />
          <ReferenceLine y={data.threshold} stroke="#972a3a" strokeDasharray="5 4"
                         label={{ value: `threshold ${pct(data.threshold)}`, ...AXIS, position: 'insideTopLeft' }} />
          {data.current_value !== null && data.current_value !== undefined && (
            <ReferenceLine x={data.current_value} stroke="#1f5fa8"
                           label={{ value: 'current', ...AXIS, position: 'top' }} />
          )}
        </ComposedChart>
      </ResponsiveContainer>
    </div>
  );
}
