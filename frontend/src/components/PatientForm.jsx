/**
 * The 8-field patient form, built entirely from /feature-info specs.
 * Used by the Prediction page and (in edit mode) by What-if analysis.
 */
const GROUPS = [
  { title: 'Clinical measurements', keys: ['glucose', 'blood_pressure', 'bmi', 'skin_thickness', 'insulin'] },
  { title: 'History and demographics', keys: ['pregnancies', 'diabetes_pedigree', 'age'] },
];

export function FormField({ spec, value, error, onChange, onBlur, disabled, idPrefix, changed }) {
  const id = `${idPrefix}-${spec.key}`;
  const hintId = `${id}-hint`;
  const errId = `${id}-err`;
  const tr = spec.training_range;
  return (
    <div className={`field${changed ? ' changed' : ''}`}>
      <label htmlFor={id}>
        <span>
          {spec.label}
          {!spec.required && <span className="optional"> (optional)</span>}
        </span>
        <span className="unit">{spec.unit}</span>
      </label>
      <input
        id={id}
        name={spec.key}
        type="text"
        inputMode={spec.integer ? 'numeric' : 'decimal'}
        autoComplete="off"
        placeholder={spec.placeholder}
        value={value ?? ''}
        onChange={(e) => onChange(spec.key, e.target.value)}
        onBlur={() => onBlur?.(spec.key)}
        disabled={disabled}
        required={spec.required}
        aria-invalid={error ? 'true' : 'false'}
        aria-describedby={error ? `${errId} ${hintId}` : hintId}
      />
      {error && <div id={errId} className="error">{error}</div>}
      <div id={hintId} className="hint">
        Accepted {spec.min_value}–{spec.max_value}
        {tr ? `; training data ${tr.min}–${tr.max}` : ''}
      </div>
    </div>
  );
}

export default function PatientForm({
  specs, values, errors = {}, onChange, onBlur, disabled = false, idPrefix = 'f', compareTo = null,
}) {
  const byKey = Object.fromEntries(specs.map((s) => [s.key, s]));
  return (
    <div className="form-groups">
      {GROUPS.map((g) => (
        <fieldset key={g.title}>
          <legend>{g.title}</legend>
          <div className="field-grid">
            {g.keys.filter((k) => byKey[k]).map((k) => (
              <FormField
                key={k}
                spec={byKey[k]}
                value={values[k]}
                error={errors[k]}
                onChange={onChange}
                onBlur={onBlur}
                disabled={disabled}
                idPrefix={idPrefix}
                changed={compareTo ? String(compareTo[k] ?? '') !== String(values[k] ?? '').trim() : false}
              />
            ))}
          </div>
        </fieldset>
      ))}
    </div>
  );
}
