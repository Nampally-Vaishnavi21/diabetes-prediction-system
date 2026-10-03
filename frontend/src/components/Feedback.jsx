/** Loading, error and empty states used on every page. */
import { Link } from 'react-router-dom';

export function Loader({ text = 'Loading…' }) {
  return <div className="loader" role="status" aria-live="polite">{text}</div>;
}

const TITLES = {
  network: 'Backend unavailable',
  timeout: 'Request timed out',
  validation: 'Please check the inputs',
  model: 'Model not available',
  bad_request: 'Request not accepted',
  not_found: 'Not found',
  server: 'Server error',
};

export function ErrorBanner({ error, onRetry, retryLabel = 'Retry' }) {
  if (!error) return null;
  return (
    <div className="banner" role="alert">
      <strong>{TITLES[error.kind] || 'Something went wrong'}</strong>
      <p>{error.message}</p>
      {onRetry && (
        <p><button type="button" className="btn" onClick={onRetry}>{retryLabel}</button></p>
      )}
    </div>
  );
}

export function NeedsPrediction({ what }) {
  return (
    <div className="empty">
      <h2>No patient analysed yet</h2>
      <p>{what} uses the patient from your most recent prediction. Enter patient information first.</p>
      <Link to="/predict" className="btn btn-primary">Go to the prediction form</Link>
    </div>
  );
}
