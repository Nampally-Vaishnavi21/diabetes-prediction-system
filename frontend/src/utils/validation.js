/**
 * Frontend validation. Rules come from GET /feature-info, so they are the same
 * limits the backend enforces with Pydantic. The backend re-checks everything;
 * this layer just gives instant feedback and stops invalid requests early.
 */

/** Returns an error message, or null if the value is valid. */
export function validateField(spec, raw) {
  const text = raw === null || raw === undefined ? '' : String(raw).trim();
  if (text === '') {
    return spec.required ? `${spec.label} is required.` : null;
  }
  // Number('') is 0 and Number('1e3') is 1000; only accept plain decimal notation
  if (!/^-?\d+(\.\d+)?$|^-?\.\d+$/.test(text)) return spec.error_message;
  const n = Number(text);
  if (!Number.isFinite(n)) return spec.error_message;
  if (spec.integer && !Number.isInteger(n)) return spec.error_message;
  if (n < spec.min_value || n > spec.max_value) return spec.error_message;
  return null;
}

export function validateAll(specs, values) {
  const errors = {};
  specs.forEach((spec) => {
    const msg = validateField(spec, values[spec.key]);
    if (msg) errors[spec.key] = msg;
  });
  return errors;
}

/** Form strings -> JSON body for the API (blank optional fields -> null). */
export function toPayload(specs, values) {
  const body = {};
  specs.forEach((spec) => {
    const text = String(values[spec.key] ?? '').trim();
    body[spec.key] = text === '' ? null : Number(text);
  });
  return body;
}

/** API patient object -> form strings. */
export function fromPatient(specs, patient) {
  const values = {};
  specs.forEach((spec) => {
    const v = patient?.[spec.key];
    values[spec.key] = v === null || v === undefined ? '' : String(v);
  });
  return values;
}

export function emptyValues(specs) {
  return Object.fromEntries(specs.map((s) => [s.key, '']));
}
