import { Link } from 'react-router-dom';

export default function NotFound() {
  return (
    <div className="empty">
      <h1>Page not found</h1>
      <p>This address does not match any page in the app.</p>
      <Link to="/" className="btn btn-primary">Go to the dashboard</Link>
    </div>
  );
}
