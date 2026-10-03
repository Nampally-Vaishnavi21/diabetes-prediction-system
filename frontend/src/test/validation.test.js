import { describe, expect, it } from 'vitest';
import { fromPatient, toPayload, validateAll, validateField } from '../utils/validation';
import { ApiError, toApiError } from '../services/api';

// Same shape as GET /feature-info returns
const AGE = { key: 'age', label: 'Age', min_value: 21, max_value: 100, integer: true, required: true,
  error_message: 'Please enter a valid age (whole number, 21–100 years).' };
const BMI = { key: 'bmi', label: 'BMI', min_value: 12, max_value: 70, integer: false, required: true,
  error_message: 'Please enter a valid BMI (12–70 kg/m²).' };
const INSULIN = { key: 'insulin', label: 'Insulin', min_value: 10, max_value: 900, integer: false, required: false,
  error_message: 'Please enter a valid insulin value (10–900 µU/mL) or leave it blank.' };
const SPECS = [AGE, BMI, INSULIN];

describe('validateField', () => {
  it('accepts valid values', () => {
    expect(validateField(AGE, '45')).toBeNull();
    expect(validateField(BMI, '28.5')).toBeNull();
  });
  it('rejects negative age and impossible BMI', () => {
    expect(validateField(AGE, '-5')).toBe(AGE.error_message);
    expect(validateField(BMI, '500')).toBe(BMI.error_message);
  });
  it('rejects decimals for integer fields, text, and exponent notation', () => {
    expect(validateField(AGE, '30.5')).toBe(AGE.error_message);
    expect(validateField(BMI, 'abc')).toBe(BMI.error_message);
    expect(validateField(BMI, '2e1')).toBe(BMI.error_message);
  });
  it('requires required fields but allows blank optional ones', () => {
    expect(validateField(AGE, '')).toBe('Age is required.');
    expect(validateField(INSULIN, '  ')).toBeNull();
  });
});

describe('payload conversion', () => {
  it('turns blank optional fields into null and numbers into numbers', () => {
    expect(toPayload(SPECS, { age: '45', bmi: '28.5', insulin: '' })).toEqual({ age: 45, bmi: 28.5, insulin: null });
  });
  it('round-trips an API patient into form strings', () => {
    expect(fromPatient(SPECS, { age: 45, bmi: 28.5, insulin: null })).toEqual({ age: '45', bmi: '28.5', insulin: '' });
  });
  it('validateAll collects every error', () => {
    expect(Object.keys(validateAll(SPECS, { age: '', bmi: '900', insulin: '' }))).toEqual(['age', 'bmi']);
  });
});

describe('toApiError', () => {
  it('maps a missing response to a backend-unavailable error', () => {
    const e = toApiError({ message: 'Network Error' });
    expect(e).toBeInstanceOf(ApiError);
    expect(e.kind).toBe('network');
    expect(e.message).toMatch(/Cannot reach the prediction server/);
  });
  it('maps timeouts', () => {
    expect(toApiError({ code: 'ECONNABORTED' }).kind).toBe('timeout');
  });
  it('maps 422 details to field errors', () => {
    const e = toApiError({ response: { status: 422, data: {
      message: 'Some inputs are missing or out of range.',
      details: [{ field: 'age', message: AGE.error_message }] } } });
    expect(e.kind).toBe('validation');
    expect(e.fieldErrors).toEqual({ age: AGE.error_message });
  });
  it('maps 503 to model-not-loaded with the server message', () => {
    const e = toApiError({ response: { status: 503, data: { message: 'Train it first' } } });
    expect(e.kind).toBe('model');
    expect(e.message).toBe('Train it first');
  });
  it('never shows server internals for 500', () => {
    const e = toApiError({ response: { status: 500, data: { message: 'Traceback (most recent call last)' } } });
    expect(e.kind).toBe('server');
    expect(e.message).not.toMatch(/Traceback/);
  });
});
