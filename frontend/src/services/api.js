/**
 * API service — the ONLY place the frontend talks to the backend.
 *
 * WHAT: Wraps every FastAPI endpoint in a small async function.
 * WHY:  Pages never build URLs themselves, so the backend address lives in one
 *       place (VITE_API_URL in frontend/.env) and every error is turned into
 *       one consistent ApiError the UI knows how to display.
 */
import axios from 'axios';

export const API_URL = (import.meta.env.VITE_API_URL || 'http://localhost:8000').replace(/\/+$/, '');
const TIMEOUT_MS = 30000;

const client = axios.create({
  baseURL: API_URL,
  timeout: TIMEOUT_MS,
  headers: { 'Content-Type': 'application/json' },
});

export class ApiError extends Error {
  /**
   * kind: 'network' | 'timeout' | 'validation' | 'model' | 'bad_request' | 'server' | 'not_found'
   * fieldErrors: { fieldPath: message } for validation errors
   */
  constructor(kind, message, { status = null, fieldErrors = {} } = {}) {
    super(message);
    this.kind = kind;
    this.status = status;
    this.fieldErrors = fieldErrors;
  }
}

/** Convert any axios error into a user-friendly ApiError (never a stack trace). */
export function toApiError(err) {
  if (err instanceof ApiError) return err;
  if (err?.code === 'ECONNABORTED' || err?.code === 'ETIMEDOUT') {
    return new ApiError('timeout',
      `The server took longer than ${TIMEOUT_MS / 1000} seconds to respond. Please try again.`);
  }
  if (!err?.response) {
    return new ApiError('network',
      `Cannot reach the prediction server at ${API_URL}. Check that the backend is running ` +
      '(uvicorn app.main:app --reload) and that VITE_API_URL is correct.');
  }
  const { status, data } = err.response;
  const serverMessage = data?.message;
  if (status === 422) {
    const fieldErrors = {};
    (data?.details || []).forEach((d) => { if (d.field) fieldErrors[d.field] = d.message; });
    return new ApiError('validation', serverMessage || 'Some inputs are invalid.', { status, fieldErrors });
  }
  if (status === 503) return new ApiError('model', serverMessage || 'The model is not available.', { status });
  if (status === 404) return new ApiError('not_found', serverMessage || 'Endpoint not found.', { status });
  if (status >= 400 && status < 500) return new ApiError('bad_request', serverMessage || 'Request rejected.', { status });
  return new ApiError('server', 'The server hit an unexpected error. Please try again.', { status });
}

async function call(promise) {
  try {
    const res = await promise;
    return res.data;
  } catch (err) {
    throw toApiError(err);
  }
}

export const api = {
  health: () => call(client.get('/health')),
  featureInfo: () => call(client.get('/feature-info')),
  modelInfo: () => call(client.get('/model-info')),
  example: () => call(client.get('/example')),
  predict: (patient) => call(client.post('/predict', patient)),
  explain: (patient) => call(client.post('/explain', patient)),
  globalExplanation: () => call(client.get('/explain/global')),
  whatIf: (original, modified) => call(client.post('/what-if', { original, modified })),
  sensitivity: (patient, feature, nPoints = 40) =>
    call(client.post('/what-if/sensitivity', { patient, feature, n_points: nPoints })),
};
