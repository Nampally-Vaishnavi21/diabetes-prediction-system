import { Link } from 'react-router-dom';
import { BeeswarmChart, ContributionTable, ImportanceChart, ShapBarChart } from '../components/charts';
import { ErrorBanner, Loader } from '../components/Feedback';
import { usePrediction } from '../context/PredictionContext';
import { useApiResource } from '../hooks/useApiResource';
import { api } from '../services/api';
import { num, pct } from '../utils/format';

export default function Explainability() {
  const { lastPatient } = usePrediction();
  const local = useApiResource(() => api.explain(lastPatient), [lastPatient], { enabled: !!lastPatient });
  const global = useApiResource(() => api.globalExplanation(), []);

  return (
    <>
      <header className="page-head">
        <h1>Explainability</h1>
        <p>
          SHAP (SHapley Additive exPlanations) splits a prediction into one contribution per input. It describes
          how the model uses the features. It does not prove that a feature causes diabetes.
        </p>
      </header>

      <section className="section" aria-labelledby="local-h">
        <h2 id="local-h">This patient</h2>
        {!lastPatient && (
          <p>
            No patient analysed yet. <Link to="/predict">Run a prediction</Link> to see a personal explanation;
            the global view below is available now.
          </p>
        )}
        {local.loading && <Loader text="Computing SHAP values for this patient…" />}
        <ErrorBanner error={local.error} onRetry={local.reload} />
        {local.data && (
          <>
            <p className="section-intro">
              The model starts from a base value of {pct(local.data.explanation.base_value, 1)} (its average output on
              background patients). Each feature then moves the estimate up or down, ending at{' '}
              {pct(local.data.probability, 1)} ({local.data.prediction_label.toLowerCase()}).
            </p>
            <ShapBarChart contributions={local.data.explanation.contributions} />
            <h3 style={{ marginTop: 'var(--space-5)' }}>Contribution table</h3>
            <ContributionTable explanation={local.data.explanation} />
            <p className="chart-caption">
              {local.data.explanation.method}. Additivity check: base {num(local.data.explanation.base_value, 4)} +
              contributions = {num(local.data.explanation.sum_check, 4)}.
            </p>
          </>
        )}
      </section>

      <section className="section" aria-labelledby="global-h">
        <h2 id="global-h">Across all test patients</h2>
        {global.loading && <Loader text="Loading global explanation…" />}
        <ErrorBanner error={global.error} onRetry={global.reload} />
        {global.data && (
          <>
            <p className="section-intro">
              Computed on {global.data.n_explained} held-out test patients. {global.data.method}.
            </p>
            <div className="grid-2">
              <div>
                <h3>Feature importance</h3>
                <p className="small muted">Average size of each feature's contribution, regardless of direction.</p>
                <ImportanceChart importance={global.data.importance} />
              </div>
              <div>
                <h3>SHAP summary</h3>
                <p className="small muted">
                  One dot per patient. Colour shows the feature value: teal = low, red = high, grey = missing.
                  Dots right of zero raised that patient's estimate.
                </p>
                <BeeswarmChart beeswarm={global.data.beeswarm} />
              </div>
            </div>
            <p className="chart-caption">{global.data.note}</p>
          </>
        )}
      </section>

      {lastPatient && (
        <div className="actions section">
          <Link to="/results" className="btn">Back to results</Link>
          <Link to="/what-if" className="btn btn-primary">Try what-if analysis</Link>
        </div>
      )}
    </>
  );
}
