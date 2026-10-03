import { useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { ShapBarChart } from '../components/charts';
import { ErrorBanner, Loader, NeedsPrediction } from '../components/Feedback';
import RiskRuler from '../components/RiskRuler';
import { usePrediction } from '../context/PredictionContext';
import { api } from '../services/api';
import { pct } from '../utils/format';

export default function Results() {
  const { result, lastPatient, saveResult } = usePrediction();
  const navigate = useNavigate();
  const [rerunning, setRerunning] = useState(false);
  const [error, setError] = useState(null);

  if (!result) return <NeedsPrediction what="The results page" />;

  const high = result.prediction === 1;
  const u = result.uncertainty;
  const agree = result.model_agreement;

  // "Run again": sends the same patient to the backend again (a real request)
  const rerun = async () => {
    setError(null);
    setRerunning(true);
    try {
      saveResult(lastPatient, await api.predict(lastPatient));
    } catch (err) {
      setError(err);
    } finally {
      setRerunning(false);
    }
  };

  return (
    <>
      <header className="page-head">
        <h1>Prediction result</h1>
        <p>Model: {result.model}</p>
      </header>

      <ErrorBanner error={error} onRetry={rerun} retryLabel="Retry" />
      {rerunning && <Loader text="Analyzing patient information..." />}

      <section aria-labelledby="readout-h">
        <h2 id="readout-h" className="visually-hidden">Model estimate</h2>
        <div className="readout">
          <div>
            <div className={`readout-value ${high ? 'high' : 'low'}`}>{pct(result.probability)}</div>
            <div className="readout-meta">model-estimated probability</div>
          </div>
          <div>
            <div className="readout-label">
              <span className={`tag ${high ? 'high' : 'low'}`}>{result.prediction_label}</span>
            </div>
            <div className="readout-meta">
              {high ? 'At or above' : 'Below'} the decision threshold of {pct(result.threshold, 1)}
            </div>
          </div>
        </div>
        <RiskRuler probability={result.probability} threshold={result.threshold} lower={u.lower} upper={u.upper}
                   label="Patient estimate" />
        <p style={{ marginTop: 'var(--space-4)' }}>{result.interpretation}</p>
      </section>

      <dl className="triad">
        <div>
          <dt>Model probability</dt>
          <dd><b>{pct(result.probability, 1)}</b>
            What the selected model outputs for these inputs. Its threshold of {pct(result.threshold, 1)} was
            chosen to balance sensitivity and specificity on training data.</dd>
        </div>
        <div>
          <dt>Model uncertainty</dt>
          <dd><b>{pct(u.lower, 1)} – {pct(u.upper, 1)}</b>
            {u.explanation}</dd>
        </div>
        <div>
          <dt>Medical diagnosis</dt>
          <dd><b>Not provided</b>
            This is a research prototype trained on 768 patients from one population. Diagnosis requires a
            clinician and laboratory tests such as HbA1c or fasting plasma glucose.</dd>
        </div>
      </dl>

      {u.borderline && (
        <div className="banner warn" role="note">
          <strong>Borderline classification</strong>
          <p>The bootstrap range crosses the threshold. {Math.round(u.share_of_models_above_threshold * 100)}% of
            the 50 bootstrap models put this patient above it.</p>
        </div>
      )}
      {result.ood_warnings.length > 0 && (
        <div className="banner warn" role="note">
          <strong>Inputs outside the training data</strong>
          {result.ood_warnings.map((w) => <p key={w.feature}>{w.message}</p>)}
        </div>
      )}
      {result.imputed_features.length > 0 && (
        <div className="banner info" role="note">
          <p>Left blank and filled in by the model's imputer (training median): {result.imputed_features.join(', ')}.</p>
        </div>
      )}

      <section className="section" aria-labelledby="drivers-h">
        <h2 id="drivers-h">What drove this estimate</h2>
        <p className="section-intro">
          SHAP contributions in probability units. Red bars raised the estimate, green bars lowered it.
          Top raising: {result.explanation.top_increasing.join(', ') || 'none'}. Top lowering:{' '}
          {result.explanation.top_decreasing.join(', ') || 'none'}.
        </p>
        <ShapBarChart contributions={result.explanation.contributions} />
        <p className="chart-caption">{result.explanation.note}</p>
      </section>

      <section className="section" aria-labelledby="agree-h">
        <h2 id="agree-h">Do the other trained models agree?</h2>
        <p className="section-intro">
          {agree.n_above_threshold} of {agree.n_models} models put this patient above the threshold. {agree.note}
        </p>
        <RiskRuler probability={result.probability} threshold={result.threshold} markers={agree.votes} compact
                   label="Model agreement" />
        <div className="table-wrap" style={{ marginTop: 'var(--space-4)' }}>
          <table>
            <thead><tr><th>Model</th><th>Probability</th><th>Side of threshold</th></tr></thead>
            <tbody>
              {agree.votes.map((v) => (
                <tr key={v.key} className={v.display_name === result.model.split(' + ')[0] ? 'selected' : ''}>
                  <td>{v.display_name}</td>
                  <td>{pct(v.probability, 1)}</td>
                  <td><span className={`tag ${v.above_threshold ? 'high' : 'low'}`}>
                    {v.above_threshold ? 'Above' : 'Below'}</span></td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      <section className="section">
        <div className="actions">
          <Link to="/explain" className="btn btn-primary">View full explanation</Link>
          <Link to="/what-if" className="btn">Try what-if analysis</Link>
          <button type="button" className="btn" onClick={() => navigate('/predict')}>Back to inputs</button>
          <button type="button" className="btn btn-quiet" onClick={rerun} disabled={rerunning}>
            {rerunning ? 'Analyzing patient information...' : 'Run prediction again'}
          </button>
        </div>
      </section>
    </>
  );
}
