import { useEffect, useState } from 'react';
import { Link, NavLink, Outlet, useLocation } from 'react-router-dom';
import { DISCLAIMER } from './Disclaimer';

const LINKS = [
  { to: '/', label: 'Dashboard', end: true },
  { to: '/predict', label: 'Prediction' },
  { to: '/results', label: 'Results' },
  { to: '/explain', label: 'Explainability' },
  { to: '/what-if', label: 'What-if analysis' },
  { to: '/model', label: 'Model information' },
  { to: '/about', label: 'About & disclaimer' },
];

export default function Layout() {
  const [open, setOpen] = useState(false);
  const location = useLocation();

  // Close the mobile menu after navigating, and move focus to the page
  useEffect(() => {
    setOpen(false);
    document.getElementById('main')?.focus({ preventScroll: true });
    window.scrollTo(0, 0);
  }, [location.pathname]);

  useEffect(() => {
    const onKey = (e) => { if (e.key === 'Escape') setOpen(false); };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, []);

  return (
    <div className="shell">
      <header className="topbar">
        <Link to="/" className="brand">Diabetes Risk Model</Link>
        <button
          type="button"
          className="btn btn-menu"
          aria-expanded={open}
          aria-controls="sidebar"
          onClick={() => setOpen((o) => !o)}
        >
          {open ? 'Close menu' : 'Menu'}
        </button>
      </header>

      <aside id="sidebar" className={`sidebar${open ? ' open' : ''}`} aria-label="Main navigation">
        <Link to="/" className="brand">
          Diabetes Risk Model
          <span>Explainable ML prototype</span>
        </Link>
        <nav className="nav">
          {LINKS.map((l) => (
            <NavLink key={l.to} to={l.to} end={l.end}>{l.label}</NavLink>
          ))}
        </nav>
        <p className="sidebar-foot">Academic prototype. Not for clinical use.</p>
      </aside>

      <div className="main">
        <div className="disclaimer-strip" role="note">{DISCLAIMER}</div>
        <main id="main" className="page" tabIndex={-1}>
          <Outlet />
        </main>
      </div>
    </div>
  );
}
