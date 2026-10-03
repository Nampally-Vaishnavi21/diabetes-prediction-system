/**
 * Shared state across pages:
 *  - featureSpecs : field definitions from GET /feature-info (loaded once)
 *  - formValues   : what is typed in the prediction form (kept when navigating away)
 *  - lastPatient  : the exact JSON body of the last successful prediction
 *  - result       : the /predict response for that patient
 */
import { createContext, useCallback, useContext, useMemo, useState } from 'react';
import { api } from '../services/api';
import { useApiResource } from '../hooks/useApiResource';

const PredictionContext = createContext(null);

export function PredictionProvider({ children }) {
  const features = useApiResource(() => api.featureInfo(), []);
  const [formValues, setFormValues] = useState({});
  const [lastPatient, setLastPatient] = useState(null);
  const [result, setResult] = useState(null);

  const saveResult = useCallback((patient, res) => {
    setLastPatient(patient);
    setResult(res);
  }, []);

  const value = useMemo(() => ({
    featureSpecs: features.data?.features || null,
    featuresLoading: features.loading,
    featuresError: features.error,
    reloadFeatures: features.reload,
    formValues,
    setFormValues,
    lastPatient,
    result,
    saveResult,
  }), [features, formValues, lastPatient, result, saveResult]);

  return <PredictionContext.Provider value={value}>{children}</PredictionContext.Provider>;
}

export function usePrediction() {
  const ctx = useContext(PredictionContext);
  if (!ctx) throw new Error('usePrediction must be used inside PredictionProvider');
  return ctx;
}
