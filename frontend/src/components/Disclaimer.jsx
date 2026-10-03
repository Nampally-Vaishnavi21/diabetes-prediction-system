export const DISCLAIMER =
  'This system is an academic/research prototype for diabetes risk prediction. It is not a medical ' +
  'diagnostic device and should not be used as a substitute for professional medical advice, ' +
  'diagnosis, or treatment.';

export default function Disclaimer() {
  return (
    <div className="banner warn" role="note">
      <strong>Medical disclaimer</strong>
      <p>{DISCLAIMER}</p>
    </div>
  );
}
