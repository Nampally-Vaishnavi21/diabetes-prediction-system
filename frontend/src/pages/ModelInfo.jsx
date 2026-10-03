import { useState } from 'react';
import {
  CalibrationChart, ConfusionMatrix, CorrelationHeatmap, MultiCurveChart,
} from '../components/charts';
import { ErrorBanner, Loader } from '../components/Feedback';
import { useApiResource } from '../hooks/useApiResource';
import { api } from '../services/api';
import { date, num, pct } from '../utils/format';

const CV_METRICS = [
  ['roc_auc', 'ROC-AUC'], ['pr_auc', 'PR-AUC'], ['accuracy', 'Accuracy'], ['precision', 'Precision'],
  ['recall', 'Recall'], ['f1', 'F1'], ['brier', 'Brier'],
];
const TEST_METRICS = [
  ['accuracy', 'Accuracy'], ['precision', 'Precision'], ['recall', 'Recall'], ['specificity', 'Specificity'],
  ['f1', 'F1'], ['roc_auc', 'ROC-AUC'], ['pr_auc', 'PR-AUC'], ['brier', 'Brier'],
];

function paramText(params) {
  return Object.entries(params)
    .map(([k, v]) => {
      const name = k === 'prep__engineer__enabled' ? 'engineered features' : k.replace(/^model__(estimator__)?/, '');
      const val = v === null ? 'none' : v === true ? 'yes' : v === false ? 'no' : Array.isArray(v) ? v.join(', ') : String(v);
      return `${name} = ${val}`;
    })
    .join('; ');
}

export default function ModelInfo() {
  const info = useApiResource(() => api.modelInfo(), []);
  const [curve, setCurve] = useState('roc');

  if (info.loading) return <Loader text="Loading model information…" />;
  if (info.error) {
    return (<><h1>Model information</h1><ErrorBanner error={info.error} onRetry={info.reload} /></>);
  }
  const d = info.data;
  const cv = d.cross_validation;
  const te = d.test_evaluation;
  const fm = te.final_model;
  const ds = d.dataset;
  const selectedBase = cv.models.find((m) => m.key === d.selected_model)?.display_name;

  const rocSeries = te.models.map((m) => ({ name: m.display_name, points: m.roc_curve, highlight: m.key === d.selected_model }));
  const prSeries = te.models.map((m) => ({ name: m.display_name, points: m.pr_curve, highlight: m.key === d.selected_model }));
  const calib = d.calibration.test_curves;

  return (
    <>
      <header className="page-head">
        <h1>Model information</h1>
        <p>Every number on this page is read from the files written by the training script on {date(d.trained_at)}.</p>
      </header>

      <section className="section" aria-labelledby="sel-h">
        <h2 id="sel-h">Selected model</h2>
        <div className="facts">
          <div><div className="v">{d.selected_model_display}</div><div className="k">Final model</div></div>
          <div><div className="v">{pct(d.threshold.value, 1)}</div><div className="k">Decision threshold</div></div>
          <div><div className="v">{ds.n_features}</div><div className="k">Input features</div></div>
          <div><div className="v">{cv.models.length}</div><div className="k">Models compared</div></div>
        </div>
        <dl className="kv">
          <dt>Why this model</dt><dd>{d.selection_reason}</dd>
          <dt>Tuned settings</dt><dd>{paramText(d.best_params)}</dd>
          <dt>Threshold method</dt><dd>{d.threshold.method}</dd>
          <dt>Calibration</dt><dd>{d.calibration.summary}</dd>
          <dt>Library</dt><dd>scikit-learn {d.sklearn_version}</dd>
        </dl>
      </section>

      <section className="section" aria-labelledby="test-h">
        <h2 id="test-h">Final model on the held-out test set</h2>
        <p className="section-intro">
          {te.test_set.n} patients ({te.test_set.n_positive} diabetic), used once after all choices were made.
          Bootstrap 95% CI for ROC-AUC: {num(fm.bootstrap.roc_auc_95ci[0], 3)}–{num(fm.bootstrap.roc_auc_95ci[1], 3)};
          Brier: {num(fm.bootstrap.brier_95ci[0], 3)}–{num(fm.bootstrap.brier_95ci[1], 3)}.
        </p>
        <div className="grid-2">
          <div className="table-wrap">
            <table>
              <thead><tr><th>Metric</th><th>Threshold {num(d.threshold.value, 3)}</th><th>Threshold 0.5</th></tr></thead>
              <tbody>
                {TEST_METRICS.map(([k, label]) => (
                  <tr key={k}><td>{label}</td><td>{num(fm.metrics_at_threshold[k], 3)}</td><td>{num(fm.metrics_at_0_5[k], 3)}</td></tr>
                ))}
              </tbody>
            </table>
          </div>
          <div>
            <h3>Confusion matrix (tuned threshold)</h3>
            <ConfusionMatrix cm={fm.metrics_at_threshold.confusion_matrix} />
          </div>
        </div>
      </section>

      <section className="section" aria-labelledby="cmp-h">
        <h2 id="cmp-h">Model comparison</h2>
        <h3>Cross-validation on the training set (mean ± std over 5 folds)</h3>
        <p className="small muted">{cv.cv_method}. {cv.selection_rule}.</p>
        <div className="table-wrap">
          <table>
            <thead><tr><th>Model</th>{CV_METRICS.map(([, l]) => <th key={l}>{l}</th>)}<th>Engineered features</th></tr></thead>
            <tbody>
              {cv.models.map((m) => (
                <tr key={m.key} className={m.key === d.selected_model ? 'selected' : ''}>
                  <td>{m.display_name}</td>
                  {CV_METRICS.map(([k]) => <td key={k}>{num(m.cv[k].mean, 3)} ± {num(m.cv[k].std, 3)}</td>)}
                  <td>{m.engineered_features === null ? '–' : m.engineered_features ? 'Yes' : 'No'}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        <h3 style={{ marginTop: 'var(--space-5)' }}>Held-out test set (threshold 0.5 for every model)</h3>
        <div className="table-wrap">
          <table>
            <thead><tr><th>Model</th>{TEST_METRICS.map(([, l]) => <th key={l}>{l}</th>)}</tr></thead>
            <tbody>
              {te.models.map((m) => (
                <tr key={m.key} className={m.key === d.selected_model ? 'selected' : ''}>
                  <td>{m.display_name}</td>
                  {TEST_METRICS.map(([k]) => <td key={k}>{num(m.test_metrics_at_0_5[k], 3)}</td>)}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <p className="chart-caption">
          The model was chosen by cross-validation before the test set was used. Choosing by test score instead would
          make the reported test results optimistic.
        </p>
      </section>

      <section className="section" aria-labelledby="curves-h">
        <h2 id="curves-h">Curves on the test set</h2>
        <div className="actions" role="tablist" aria-label="Curve type" style={{ marginBottom: 'var(--space-4)' }}>
          {[['roc', 'ROC'], ['pr', 'Precision–recall'], ['cal', 'Calibration']].map(([k, l]) => (
            <button key={k} type="button" role="tab" aria-selected={curve === k}
                    className={`btn ${curve === k ? 'btn-primary' : ''}`} onClick={() => setCurve(k)}>{l}</button>
          ))}
        </div>
        {curve === 'roc' && (
          <MultiCurveChart series={rocSeries} xKey="fpr" yKey="tpr" diagonal
                           xLabel="False positive rate" yLabel="True positive rate" />
        )}
        {curve === 'pr' && (
          <MultiCurveChart series={prSeries} xKey="recall" yKey="precision"
                           baseline={te.test_set.positive_rate} xLabel="Recall" yLabel="Precision" />
        )}
        {curve === 'cal' && (
          <>
            <CalibrationChart series={[
              { name: `${selectedBase} (Brier ${num(calib.uncalibrated.brier, 3)})`, points: calib.uncalibrated.points },
              { name: `With extra sigmoid layer (Brier ${num(calib.calibrated.brier, 3)})`, points: calib.calibrated.points },
            ]} />
            <p className="chart-caption">
              CV Brier without the extra layer {num(d.calibration.cv_brier_uncalibrated.mean, 4)}, with it{' '}
              {num(d.calibration.cv_brier_calibrated.mean, 4)}. {d.calibration.decision} Bins: {d.calibration.n_bins} ({d.calibration.binning}).
            </p>
          </>
        )}
      </section>

      {te.uncertainty && (
        <section className="section" aria-labelledby="unc-h">
          <h2 id="unc-h">Uncertainty method</h2>
          <p>
            {te.uncertainty.n_models} bootstrap models are trained on resampled training data. For each patient the
            app reports the {te.uncertainty.interval} of their estimates. On the test set the average range width was{' '}
            {pct(te.uncertainty.test_mean_interval_width, 1)}, and {pct(te.uncertainty.test_share_borderline, 1)} of
            patients were borderline (range crossing the threshold).
          </p>
        </section>
      )}

      <section className="section" aria-labelledby="data-h">
        <h2 id="data-h">Dataset</h2>
        <dl className="kv">
          <dt>Name</dt><dd>{ds.name}</dd>
          <dt>Origin</dt><dd>{ds.origin}</dd>
          <dt>Population</dt><dd>{ds.population}</dd>
          <dt>Licence</dt><dd>{ds.license}</dd>
          <dt>Source</dt><dd><a href={ds.url} target="_blank" rel="noreferrer">{ds.url}</a></dd>
          <dt>Patients</dt><dd>{ds.n_rows} ({ds.n_train} training, {ds.n_test} test); {ds.n_duplicates} duplicates</dd>
          <dt>Outcome</dt>
          <dd>{ds.class_counts.diabetic_1} diabetic, {ds.class_counts.non_diabetic_0} non-diabetic ({num(ds.positive_rate_pct, 1)}% positive)</dd>
        </dl>

        <h3 style={{ marginTop: 'var(--space-5)' }}>Missing values (recorded as 0 in the file)</h3>
        <div className="table-wrap">
          <table>
            <thead><tr><th>Feature</th><th>Missing</th><th>Share</th><th></th></tr></thead>
            <tbody>
              {Object.entries(ds.zero_encoded_missing).map(([col, v]) => (
                <tr key={col}>
                  <td>{col}</td><td>{v.count}</td><td>{num(v.pct, 1)}%</td>
                  <td style={{ width: '40%' }}>
                    <div style={{ background: 'var(--rule)', height: 8, borderRadius: 4 }}>
                      <div style={{ width: `${v.pct}%`, background: 'var(--ink)', height: 8, borderRadius: 4 }} />
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        <h3 style={{ marginTop: 'var(--space-5)' }}>Feature statistics</h3>
        <div className="table-wrap">
          <table>
            <thead>
              <tr><th>Feature</th><th>Valid</th><th>Median</th><th>Min</th><th>Max</th>
                <th>Mean, diabetic</th><th>Mean, non-diabetic</th><th>IQR outliers</th></tr>
            </thead>
            <tbody>
              {Object.entries(ds.feature_stats).map(([k, s]) => (
                <tr key={k}>
                  <td>{s.column}</td><td>{s.count_valid}</td><td>{num(s.median, 2)}</td><td>{num(s.min, 2)}</td>
                  <td>{num(s.max, 2)}</td><td>{num(s.mean_diabetic, 2)}</td><td>{num(s.mean_non_diabetic, 2)}</td>
                  <td>{s.outliers_iqr.n_outliers}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <p className="chart-caption">{ds.outlier_policy}</p>

        <h3 style={{ marginTop: 'var(--space-5)' }}>Correlation matrix</h3>
        <CorrelationHeatmap matrix={ds.correlation_matrix} />
        <p className="chart-caption">Pearson correlation; red positive, blue negative. Missing values excluded pairwise.</p>
      </section>

      <section className="section" aria-labelledby="fs-h">
        <h2 id="fs-h">Feature selection</h2>
        <p>{d.feature_selection.decision}</p>
        <div className="table-wrap">
          <table>
            <thead><tr><th>Column after preprocessing</th><th>ANOVA F</th><th>Mutual information</th><th>RFECV rank</th></tr></thead>
            <tbody>
              {d.feature_selection.columns_after_preprocessing.map((c) => (
                <tr key={c}>
                  <td>{c}</td><td>{num(d.feature_selection.anova_f[c], 2)}</td>
                  <td>{num(d.feature_selection.mutual_information[c], 4)}</td>
                  <td>{d.feature_selection.rfecv.ranking[c]}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <p className="chart-caption">{d.feature_selection.method}.</p>
      </section>

      <section className="section" aria-labelledby="meth-h">
        <h2 id="meth-h">Training methodology</h2>
        <ol className="steps">{d.methodology.map((s) => <li key={s}>{s}</li>)}</ol>
      </section>

      <section className="section" aria-labelledby="lim-h">
        <h2 id="lim-h">Limitations</h2>
        <ul className="steps">{d.limitations.map((s) => <li key={s}>{s}</li>)}</ul>
      </section>
    </>
  );
}
