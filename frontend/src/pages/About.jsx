import { Link } from 'react-router-dom';
import Disclaimer from '../components/Disclaimer';

export default function About() {
  return (
    <>
      <header className="page-head">
        <h1>About this system</h1>
        <p>An academic project demonstrating an end-to-end, explainable machine-learning pipeline.</p>
      </header>

      <Disclaimer />

      <section className="section" aria-labelledby="read-h">
        <h2 id="read-h">How to read a result</h2>
        <p>
          A result such as 62% means: the model estimates a probability of approximately 62% based on the input
          features. It does not mean the patient has a 62% chance of having diabetes in a medical sense. The
          estimate reflects patterns in a small research dataset and inherits its limitations.
        </p>
        <p>
          The label ("higher" or "lower estimated risk") only says which side of the model's decision threshold the
          estimate falls on. The threshold was chosen to balance sensitivity and specificity on training data, so
          it is lower than 50%.
        </p>
        <p>
          The bootstrap range shows how much the estimate moves when the model is retrained on resampled data.
          SHAP contributions show how the model used each input. Neither is a statement about medical causes.
        </p>
      </section>

      <section className="section" aria-labelledby="lim-h">
        <h2 id="lim-h">Limitations</h2>
        <ul className="steps">
          <li>The training data covers 768 women aged 21 or older of Pima heritage in Arizona. The model is not valid
            for men, children or other populations.</li>
          <li>Insulin and skin thickness are missing for a large share of the dataset and are imputed.</li>
          <li>Test results come from 154 patients, so metrics carry wide confidence intervals.</li>
          <li>What-if results describe the model's behaviour, not the effect of a treatment or lifestyle change.</li>
          <li>The system has not been clinically validated and must not inform any medical decision.</li>
        </ul>
      </section>

      <section className="section" aria-labelledby="tech-h">
        <h2 id="tech-h">How it is built</h2>
        <dl className="kv">
          <dt>Frontend</dt><dd>React (Vite), React Router, Recharts, axios</dd>
          <dt>Backend</dt><dd>Python FastAPI with Pydantic validation</dd>
          <dt>Machine learning</dt><dd>scikit-learn pipelines, GridSearchCV, SHAP, bootstrap ensemble</dd>
          <dt>Dataset</dt><dd>Pima Indians Diabetes Database (NIDDK), CC0 licence</dd>
        </dl>
        <div className="actions" style={{ marginTop: 'var(--space-5)' }}>
          <Link to="/model" className="btn">See model details</Link>
          <Link to="/predict" className="btn btn-primary">Start a prediction</Link>
        </div>
      </section>
    </>
  );
}
