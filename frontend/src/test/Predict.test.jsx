/**
 * Component test for the prediction form. The API module is replaced with
 * test doubles here ONLY so the form logic can be tested without a server;
 * the real backend integration is covered by e2e/run_e2e.py.
 */
import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { MemoryRouter, Route, Routes } from 'react-router-dom';
import { beforeEach, describe, expect, it, vi } from 'vitest';

const SPECS = [
  ['pregnancies', 'Pregnancies', 0, 20, true, true], ['glucose', 'Glucose', 40, 400, false, true],
  ['blood_pressure', 'Blood Pressure (diastolic)', 20, 140, false, true],
  ['skin_thickness', 'Skin Thickness (triceps)', 5, 100, false, false],
  ['insulin', 'Insulin (2-hour serum)', 10, 900, false, false], ['bmi', 'BMI', 12, 70, false, true],
  ['diabetes_pedigree', 'Diabetes Pedigree Function', 0.05, 3, false, true], ['age', 'Age', 21, 100, true, true],
].map(([key, label, min, max, integer, required]) => ({
  key, label, unit: '', min_value: min, max_value: max, integer, required, step: 1,
  placeholder: '', description: '', error_message: `Please enter a valid ${label}.`, training_range: null,
}));

vi.mock('../services/api', async () => {
  const actual = await vi.importActual('../services/api');
  return {
    ...actual,
    api: {
      featureInfo: vi.fn(),
      predict: vi.fn(),
      example: vi.fn(),
    },
  };
});

const { api, ApiError } = await import('../services/api');
const { PredictionProvider } = await import('../context/PredictionContext');
const { default: Predict } = await import('../pages/Predict');

function renderPage() {
  return render(
    <MemoryRouter initialEntries={['/predict']} future={{ v7_startTransition: true, v7_relativeSplatPath: true }}>
      <PredictionProvider>
        <Routes>
          <Route path="/predict" element={<Predict />} />
          <Route path="/results" element={<h1>Results page</h1>} />
        </Routes>
      </PredictionProvider>
    </MemoryRouter>,
  );
}

const VALID = { pregnancies: '2', glucose: '150', blood_pressure: '70', bmi: '32', diabetes_pedigree: '0.35', age: '45' };

async function fill(user, values) {
  for (const [k, v] of Object.entries(values)) {
    const input = document.querySelector(`#predict-${k}`);
    await user.clear(input);
    await user.type(input, v);
  }
}

beforeEach(() => {
  vi.clearAllMocks();
  api.featureInfo.mockResolvedValue({ features: SPECS, target: 'Outcome', note: '' });
});

describe('Predict page', () => {
  it('blocks submission and shows errors when required fields are empty', async () => {
    const user = userEvent.setup();
    renderPage();
    await user.click(await screen.findByRole('button', { name: 'Predict' }));
    expect(screen.getByText('6 fields need correcting before the prediction can run.')).toBeInTheDocument();
    expect(screen.getByText('Glucose is required.')).toBeInTheDocument();
    expect(api.predict).not.toHaveBeenCalled();
  });

  it('sends a numeric payload with blank optional fields as null, then opens results', async () => {
    const user = userEvent.setup();
    api.predict.mockResolvedValue({ probability: 0.5 });
    renderPage();
    await screen.findByRole('button', { name: 'Predict' });
    await fill(user, VALID);
    await user.click(screen.getByRole('button', { name: 'Predict' }));
    await screen.findByText('Results page');
    expect(api.predict).toHaveBeenCalledWith({
      pregnancies: 2, glucose: 150, blood_pressure: 70, skin_thickness: null, insulin: null,
      bmi: 32, diabetes_pedigree: 0.35, age: 45,
    });
  });

  it('shows the backend error, re-enables the button and offers Retry', async () => {
    const user = userEvent.setup();
    api.predict.mockRejectedValueOnce(new ApiError('network', 'Cannot reach the prediction server.'));
    renderPage();
    await screen.findByRole('button', { name: 'Predict' });
    await fill(user, VALID);
    await user.click(screen.getByRole('button', { name: 'Predict' }));
    expect(await screen.findByText('Backend unavailable')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Predict' })).toBeEnabled();
    api.predict.mockResolvedValueOnce({ probability: 0.5 });
    await user.click(screen.getByRole('button', { name: 'Retry prediction' }));
    await screen.findByText('Results page');
  });

  it('Reset form clears the fields', async () => {
    const user = userEvent.setup();
    renderPage();
    await screen.findByRole('button', { name: 'Predict' });
    await fill(user, { glucose: '150' });
    await user.click(screen.getByRole('button', { name: 'Reset form' }));
    await waitFor(() => expect(document.querySelector('#predict-glucose').value).toBe(''));
  });

  it('shows a retry button when the form definition cannot load', async () => {
    api.featureInfo.mockRejectedValueOnce(new ApiError('network', 'Cannot reach the prediction server.'));
    renderPage();
    expect(await screen.findByRole('button', { name: 'Retry loading the form' })).toBeInTheDocument();
  });
});
