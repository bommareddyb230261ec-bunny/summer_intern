import { useEffect, useState } from "react";
import {
  BrowserRouter,
  Navigate,
  Route,
  Routes,
  useNavigate,
} from "react-router-dom";
import api from "./api/api";
import Dashboard from "./Dashboard";
import "./Login.css";

function LoginPage() {
  const [user, setUser] = useState(null);
  const [error, setError] = useState("");
  const navigate = useNavigate();

  useEffect(() => {
    const params = new URLSearchParams(window.location.search);
    const token = params.get("access_token");

    if (token) {
      localStorage.setItem("access_token", token);
      window.history.replaceState({}, "", window.location.pathname);
      navigate("/dashboard", { replace: true });
      return;
    }

    const storedToken = localStorage.getItem("access_token");
    if (!storedToken) {
      return;
    }

    api
      .get("/profile")
      .then(({ data }) => {
        setUser(data);
        navigate("/dashboard", { replace: true });
      })
      .catch(() => {
        localStorage.removeItem("access_token");
        setUser(null);
      });
  }, [navigate]);

  const handleLogin = () => {
    const apiUrl = import.meta.env.VITE_API_URL || "http://localhost:8000";
    window.location.assign(`${apiUrl}/login`);
  };

  return (
    <main className="login-shell">
      <div className="login-overlay" />

      <section className="login-card">
        <div className="login-card__eyebrow">NSG AI SURVEILLANCE DASHBOARD</div>
        <h1 className="login-card__title">Welcome Back</h1>
        <p className="login-card__subtitle">
          Secure AI-Powered Surveillance Platform
        </p>

        <button className="google-button" onClick={handleLogin}>
          <span className="google-button__icon" aria-hidden="true">
            <svg
              viewBox="0 0 24 24"
              fill="none"
              xmlns="http://www.w3.org/2000/svg"
            >
              <path
                d="M22.5 12.24c0-.78-.07-1.54-.2-2.27H12v4.3h5.96c-.26 1.4-1.03 2.58-2.18 3.37v2.8h3.52c2.06-1.9 3.25-4.7 3.25-8.2Z"
                fill="#4285F4"
              />
              <path
                d="M12 23c2.97 0 5.46-.98 7.28-2.67l-3.52-2.8c-.98.66-2.24 1.04-3.76 1.04-2.9 0-5.35-1.96-6.23-4.6H1.94v2.88C3.76 20.83 7.57 23 12 23Z"
                fill="#34A853"
              />
              <path
                d="M5.77 14.97A7.41 7.41 0 0 1 5.4 12c0-.96.17-1.89.47-2.77V6.35H1.94A11.96 11.96 0 0 0 1 12c0 1.92.46 3.73 1.27 5.35l3.5-2.38Z"
                fill="#FBBC05"
              />
              <path
                d="M12 4.77c1.62 0 3.07.56 4.21 1.65l3.16-3.16C17.44 1.37 14.97 0 12 0 7.57 0 3.76 2.17 1.94 5.35l3.5 2.88C6.65 6.73 9.1 4.77 12 4.77Z"
                fill="#EA4335"
              />
            </svg>
          </span>
          Sign in with Google
        </button>

        {error ? <p className="login-card__error">{error}</p> : null}
      </section>
    </main>
  );
}

function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<LoginPage />} />
        <Route path="/auth/callback" element={<LoginPage />} />
        <Route path="/dashboard" element={<Dashboard />} />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </BrowserRouter>
  );
}

export default App;
