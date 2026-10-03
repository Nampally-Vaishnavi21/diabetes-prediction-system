import { useEffect, useMemo, useState } from 'react';
import { Link } from 'react-router-dom';
import { SensitivityChart } from '../components/charts';
import { ErrorBanner, Loader, NeedsPrediction } from '../components/Feedback';
import PatientForm from '../components/PatientForm';
import RiskRuler from '../components/RiskRuler';
import { usePrediction } from '../context/PredictionContext';
import { useApiResource } from '../hooks/useApiResource';
import { api } from '../services/api';
import { pct, pp } from '../utils/format';
import { fromPatient, toPayload, validateAll, validateField } from '../utils/validation';

export default function WhatIf() {
  const { lastPatient, featureSpecs: specs } = usePrediction();
  const original = useMemo(
    () => (specs && lastPatient ? fromPatient(specs, lastPatient) : null), [specs, lastPatient]);

  const [values, setValues] = useState(original || {});
  const [errors, setErrors] = useState({});
  const [running, setRunning] = useState(false);
  const [error, setError] = useState(null);
  const [comparison, setComparison] = useState(null);
  const [feature, setFeature] = useState('glucose');
  const [curveScenario, setCurveScenario] = useState('original');

  useEffect(() => { if (original) setValues(original); }, [original]);

  const curvePatient = curveScenario === 'modified' && comparison ? comparison.modifiedPayload : lastPatient;
  const curve = useApiResource(() => api.sensitivity(curvePatient, feature, 40),
    [curvePatient, feature], { enabled: !!curvePatient });

  if (!lastPatient || !specs) return <NeedsPrediction what="What-if analysis" />;

  const changedKeys = specs.filter((s) => String(values[s.key] ?? '').trim() !== String(original[s.key] ?? '')).map((s) => s.key);

  const onChange = (key, v) => {
    setValues((prev) => ({ ...prev, [key]: v }));
    if (errors[key]) setErrors((e) => ({ ...e, [key]: undefined }));
  };
  const onBlur = (key) => {
    const msg = validateField(specs.find((s) => s.key === key), values[key]);
    setErrors((e) => ({ ...e, [key]: msg || undefined }));
  };

  const run = async (e) => {
    e?.preventDefault();
    setError(null);
    const errs = validateAll(specs, values);
    setErrors(errs);
    if (Object.keys(errs).length) return;
    const modifiedPayload = toPayload(specs, values);
    setRunning(true);
    try {
      const res = await api.whatIf(lastPatient, modifiedPayload);
      setComparison({ ...res, modifiedPayload });
    } catch (err) {
      setError(err);
      if (err.kind === 'validation') {
        const mapped = {};
        Object.entries(err.fieldErrors).forEach(([k, v]) => { mapped[k.replace(/^modified\./, '')] = v; });
        setErrors(mapped);
      }
    } finally {
      setRunning(false);
    }
  };

  const resetChanges = () => {
    setValues(original);
    setErrors({});
    setError(null);
    setComparison(null);
    setCurveScenario('original');
  };

  const delta = comparison?.percentage_point_change;

  return (
    <>
      <header className="page-head">
        <h1>What-if analysis</h1>
        <p>
          Change some inputs and see how the model's output responds. This is model sensitivity analysis: it shows
          what the model does, not what would happen to a real patient whose values changed.
        </p>
      </header>

      <form onSubmit={run} noValidate>
        <div className="compare">
          <div>
            <h2>Original</h2>
            <p className="small muted">From your last prediction (read-only).</p>
            <PatientForm specs={specs} values={original} onChange={() => {}} disabled idPrefix="orig" />
          </div>
          <div>
            <h2>What-if scenario</h2>
            <p className="small muted">Edit any value. Changed fields are marked in blue.</p>
            <PatientForm specs={specs} values={values} errors={errors} onChange={onChange} onBlur={onBlur}
                         disabled={running} idPrefix="whatif" compareTo={original} />
          </div>
        </div>

        <ErrorBanner error={error} onRetry={error && error.kind !== 'validation' ? run : null} />
        <div className="actions" style={{ marginTop: 'var(--space-5)' }}>
          <button type="submit" className="btn btn-primary" disabled={running || changedKeys.length === 0}>
            {running ? 'Analyzing patient information...' : 'Run comparison'}
          </button>
          <button type="button" className="btn" onClick={resetChanges} disabled={running}>Reset changes</button>
          <Link to="/results" className="btn btn-quiet">Back to results</Link>
          {changedKeys.length === 0 && <span className="small muted">Change at least one value to compare.</span>}
        </div>
        {running && <Loader text="Analyzing patient information..." />}
      </form>

      {comparison && (
        <section className="section" aria-labelledby="cmp-h" aria-live="polite">
          <h2 id="cmp-h">Comparison</h2>
          <div className="facts">
            <div><div className="v">{pct(comparison.original.probability, 1)}</div><div className="k">Original probability</div></div>
            <div><div className="v">{pct(comparison.modified.probability, 1)}</div><div className="k">What-if probability</div></div>
            <div>
              <div className={`delta ${delta > 0 ? 'up' : delta < 0 ? 'down' : ''}`}>{pp(delta)}</div>
              <div className="k">Difference (percentage points)</div>
            </div>
          </div>
          <h3>Original</h3>
          <RiskRuler probability={comparison.original.probability} threshold={comparison.threshold}
                     lower={comparison.original.uncertainty_lower} upper={comparison.original.uncertainty_upper}
                     compact label="Original" />
          <h3>What-if scenario</h3>
          <RiskRuler probability={comparison.modified.probability} threshold={comparison.threshold}
                     lower={comparison.modified.uncertainty_lower} upper={comparison.modified.uncertainty_upper}
                     compact label="What-if" />
          <p>
            {comparison.prediction_changed
              ? `The classification changes from "${comparison.original.prediction_label}" to "${comparison.modified.prediction_label}".`
              : `The classification stays "${comparison.modified.prediction_label}".`}
          </p>
          <div className="table-wrap">
            <table>
              <thead><tr><th>Changed input</th><th>Original</th><th>What-if</th><th>SHAP before</th><th>SHAP after</th></tr></thead>
              <tbody>
                {comparison.changed_features.map((c) => {
                  const before = comparison.original.explanation.contributions.find((x) => x.feature === c.feature);
                  const after = comparison.modified.explanation.contributions.find((x) => x.feature === c.feature);
                  return (
                    <tr key={c.feature}>
                      <td>{c.label}</td>
                      <td>{c.original ?? 'blank'}</td>
                      <td>{c.modified ?? 'blank'}</td>
                      <td>{before.shap_value.toFixed(4)}</td>
                      <td>{after.shap_value.toFixed(4)}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
          <p className="chart-caption">{comparison.note}</p>
        </section>
      )}

      <section className="section" aria-labelledby="curve-h">
        <h2 id="curve-h">Sensitivity curve</h2>
        <p className="section-intro">
          Vary one input across its training-data range while holding the others fixed.
        </p>
        <div className="actions" style={{ marginBottom: 'var(--space-4)' }}>
          <label htmlFor="feature-select" className="small"><b>Input to vary</b></label>
          <select id="feature-select" className="select" value={feature} onChange={(e) => setFeature(e.target.value)}>
            {specs.map((s) => <option key={s.key} value={s.key}>{s.label}</option>)}
          </select>
          {comparison && (
            <>
              <label htmlFor="scenario-select" className="small"><b>Scenario</b></label>
              <select id="scenario-select" className="select" value={curveScenario}
                      onChange={(e) => setCurveScenario(e.target.value)}>
                <option value="original">Original</option>
                <option value="modified">What-if</option>
              </select>
            </>
          )}
        </div>
        {curve.loading && <Loader text="Computing the sensitivity curve…" />}
        <ErrorBanner error={curve.error} onRetry={curve.reload} />
        {curve.data && !curve.loading && (
          <>
            <SensitivityChart data={curve.data} />
            <p className="chart-caption">{curve.data.range_source}. {curve.data.note}</p>
          </>
        )}
      </section>
    </>
  );
}
