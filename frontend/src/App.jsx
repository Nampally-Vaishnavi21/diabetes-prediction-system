import { Route, Routes } from 'react-router-dom';
import Layout from './components/Layout';
import About from './pages/About';
import Dashboard from './pages/Dashboard';
import Explainability from './pages/Explainability';
import ModelInfo from './pages/ModelInfo';
import NotFound from './pages/NotFound';
import Predict from './pages/Predict';
import Results from './pages/Results';
import WhatIf from './pages/WhatIf';

export default function App() {
  return (
    <Routes>
      <Route element={<Layout />}>
        <Route index element={<Dashboard />} />
        <Route path="predict" element={<Predict />} />
        <Route path="results" element={<Results />} />
        <Route path="explain" element={<Explainability />} />
        <Route path="what-if" element={<WhatIf />} />
        <Route path="model" element={<ModelInfo />} />
        <Route path="about" element={<About />} />
        <Route path="*" element={<NotFound />} />
      </Route>
    </Routes>
  );
}
