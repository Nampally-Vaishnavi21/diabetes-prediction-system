import { useRef, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { ErrorBanner, Loader } from '../components/Feedback';
import PatientForm from '../components/PatientForm';
import { usePrediction } from '../context/PredictionContext';
import { api } from '../services/api';
import { emptyValues, fromPatient, toPayload, validateAll, validateField } from '../utils/validation';

export default function Predict() {
  const {
    featureSpecs: specs, featuresLoading, featuresError, reloadFeatures,
    formValues, setFormValues, saveResult,
  } = usePrediction();
  const navigate = useNavigate();
  const [errors, setErrors] = useState({});
  const [submitting, setSubmitting] = useState(false);
  const [loadingExample, setLoadingExample] = useState(false);
  const [apiError, setApiError] = useState(null);
  const [notice, setNotice] = useState(null);
  const formRef = useRef(null);

  if (featuresLoading) return <Loader text="Loading the patient form…" />;
  if (featuresError || !specs) {
    return (
      <>
        <h1>Diabetes prediction</h1>
        <ErrorBanner error={featuresError} onRetry={reloadFeatures} retryLabel="Retry loading the form" />
      </>
    );
  }

  const values = { ...emptyValues(specs), ...formValues };
  const busy = submitting || loadingExample;

  const onChange = (key, v) => {
    setFormValues((prev) => ({ ...prev, [key]: v }));
    if (errors[key]) setErrors((e) => ({ ...e, [key]: undefined }));
  };

  const onBlur = (key) => {
    const spec = specs.find((s) => s.key === key);
    const msg = validateField(spec, values[key]);
    setErrors((e) => ({ ...e, [key]: msg || undefined }));
  };

  const focusFirstError = (errs) => {
    const first = specs.find((s) => errs[s.key]);
    if (first) formRef.current?.querySelector(`[name="${first.key}"]`)?.focus();
  };

  const submit = async (e) => {
    e?.preventDefault();
    setApiError(null);
    setNotice(null);
    const errs = validateAll(specs, values);
    setErrors(errs);
    if (Object.keys(errs).length) {
      focusFirstError(errs);
      return;
    }
    const payload = toPayload(specs, values);
    setSubmitting(true);
    try {
      const res = await api.predict(payload);
      saveResult(payload, res);
      navigate('/results');
    } catch (err) {
      setApiError(err);
      if (err.kind === 'validation') {
        setErrors(err.fieldErrors);
        focusFirstError(err.fieldErrors);
      }
    } finally {
      setSubmitting(false);
    }
  };

  const loadExample = async () => {
    setApiError(null);
    setLoadingExample(true);
    try {
      const ex = await api.example();
      setFormValues(fromPatient(specs, ex.patient));
      setErrors({});
      setNotice(`Loaded a real patient from the held-out test set. Recorded outcome in the dataset: ` +
        `${ex.recorded_outcome === 1 ? 'diabetic (1)' : 'non-diabetic (0)'}. ${ex.source}.`);
    } catch (err) {
      setApiError(err);
    } finally {
      setLoadingExample(false);
    }
  };

  const reset = () => {
    setFormValues(emptyValues(specs));
    setErrors({});
    setApiError(null);
    setNotice(null);
    formRef.current?.querySelector('input')?.focus();
  };

  const nErrors = Object.values(errors).filter(Boolean).length;

  return (
    <>
      <header className="page-head">
        <h1>Diabetes prediction</h1>
        <p>Enter the patient's measurements. Insulin and skin thickness can be left blank if they were not measured.</p>
      </header>

      {notice && <div className="banner info" role="status"><p>{notice}</p></div>}
      <ErrorBanner error={apiError} onRetry={apiError && apiError.kind !== 'validation' ? submit : null}
                   retryLabel="Retry prediction" />
      {nErrors > 0 && !apiError && (
        <div className="banner" role="alert">
          <p>{nErrors === 1 ? '1 field needs' : `${nErrors} fields need`} correcting before the prediction can run.</p>
        </div>
      )}

      <form ref={formRef} onSubmit={submit} noValidate className="panel" aria-busy={submitting}>
        <PatientForm specs={specs} values={values} errors={errors} onChange={onChange} onBlur={onBlur}
                     disabled={submitting} idPrefix="predict" />
        <div className="actions" style={{ marginTop: 'var(--space-5)' }}>
          <button type="submit" className="btn btn-primary" disabled={busy}>
            {submitting ? 'Analyzing patient information...' : 'Predict'}
          </button>
          <button type="button" className="btn" onClick={loadExample} disabled={busy}>
            {loadingExample ? 'Loading example…' : 'Load example'}
          </button>
          <button type="button" className="btn btn-quiet" onClick={reset} disabled={submitting}>
            Reset form
          </button>
        </div>
        {submitting && <Loader text="Analyzing patient information..." />}
      </form>
    </>
  );
}
