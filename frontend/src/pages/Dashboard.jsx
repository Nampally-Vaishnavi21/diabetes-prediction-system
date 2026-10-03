import { Link } from 'react-router-dom';
import { ErrorBanner, Loader } from '../components/Feedback';
import RiskRuler from '../components/RiskRuler';
import { usePrediction } from '../context/PredictionContext';
import { useApiResource } from '../hooks/useApiResource';
import { api } from '../services/api';
import { date, num, pct } from '../utils/format';

const PIPELINE = [
  'You enter eight clinical values; the form checks them against the same limits the server uses.',
  'The FastAPI backend validates them again and passes the raw values to the saved scikit-learn pipeline.',
  'The pipeline treats impossible zeros as missing, imputes them, scales the features and applies the trained SVM.',
  'The probability is compared with a decision threshold chosen on training data only.',
  '50 bootstrap models measure how stable the estimate is; SHAP explains which inputs drove it.',
];

export default function Dashboard() {
  const health = useApiResource(() => api.health(), []);
  const info = useApiResource(() => api.modelInfo(), [], { enabled: !!health.data?.model_loaded });
  const { result } = usePrediction();

  const fm = info.data?.test_evaluation?.final_model;
  const m = fm?.metrics_at_threshold;

  return (
    <>
      <header className="page-head">
        <h1>Diabetes risk prediction</h1>
        <p>
          A machine-learning model trained on the Pima Indians Diabetes Database estimates the probability
          of a diabetes outcome from eight clinical measurements, and explains how it reached that estimate.
        </p>
      </header>

      <section className="section" aria-labelledby="status-h">
        <h2 id="status-h">System status</h2>
        {health.loading && <Loader text="Checking the backend…" />}
        <ErrorBanner error={health.error} onRetry={health.reload} retryLabel="Check again" />
        {health.data && (
          <dl className="kv">
            <dt>Backend</dt>
            <dd><span className="status-dot ok" />Connected (API v{health.data.api_version})</dd>
            <dt>Model</dt>
            <dd>
              <span className={`status-dot ${health.data.model_loaded ? 'ok' : 'bad'}`} />
              {health.data.model_loaded ? health.data.model_name : 'Not loaded'}
            </dd>
            {health.data.trained_at && (<><dt>Trained</dt><dd>{date(health.data.trained_at)}</dd></>)}
            {!health.data.model_loaded && (<><dt>Action needed</dt><dd>{health.data.message}</dd></>)}
          </dl>
        )}
      </section>

      {health.data?.model_loaded && (
        <section className="section" aria-labelledby="perf-h">
          <h2 id="perf-h">Performance on held-out patients</h2>
          <p className="section-intro">
            Measured once on {info.data?.test_evaluation?.test_set?.n ?? '…'} patients the model never saw
            during training or tuning.
          </p>
          {info.loading && <Loader text="Loading model results…" />}
          <ErrorBanner error={info.error} onRetry={info.reload} />
          {m && (
            <div className="facts">
              <div><div className="v">{num(m.roc_auc, 3)}</div>
                <div className="k">ROC-AUC (95% CI {num(fm.bootstrap.roc_auc_95ci[0], 2)}–{num(fm.bootstrap.roc_auc_95ci[1], 2)})</div></div>
              <div><div className="v">{pct(m.recall)}</div><div className="k">Recall (sensitivity)</div></div>
              <div><div className="v">{pct(m.specificity)}</div><div className="k">Specificity</div></div>
              <div><div className="v">{num(m.brier, 3)}</div><div className="k">Brier score</div></div>
              <div><div className="v">{pct(info.data.threshold.value, 1)}</div><div className="k">Decision threshold</div></div>
            </div>
          )}
          <div className="actions">
            <Link to="/predict" className="btn btn-primary">Start a prediction</Link>
            <Link to="/model" className="btn">See full model results</Link>
          </div>
        </section>
      )}

      {result && (
        <section className="section" aria-labelledby="last-h">
          <h2 id="last-h">Most recent prediction</h2>
          <p>
            {result.prediction_label}: the model estimates approximately {pct(result.probability)}.
          </p>
          <RiskRuler probability={result.probability} threshold={result.threshold}
                     lower={result.uncertainty.lower} upper={result.uncertainty.upper} compact />
          <div className="actions">
            <Link to="/results" className="btn">Open results</Link>
          </div>
        </section>
      )}

      <section className="section" aria-labelledby="how-h">
        <h2 id="how-h">What happens when you press Predict</h2>
        <ol className="steps">{PIPELINE.map((s) => <li key={s}>{s}</li>)}</ol>
      </section>
    </>
  );
}
