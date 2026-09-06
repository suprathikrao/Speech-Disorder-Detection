import { useState } from 'react';
import './LoginPage.css';

// Preset Verified Clinical Demonstration Profiles
const DEMO_PROFILES = [
  {
    id: 'lead-slp',
    name: 'Dr. Suprathik Rao',
    role: 'Chief Speech-Language Pathologist & Lead Researcher',
    email: 'dr.suprathik@acoustiscreen.health',
    initials: 'SR',
    badgeColor: '#0ea5e9',
    badge: 'Clinical Administrator'
  },
  {
    id: 'voice-clinician',
    name: 'Sarah Chen, M.S. CCC-SLP',
    role: 'Senior Voice & Neuro-Pathology Examiner',
    email: 's.chen@voicepathology.org',
    initials: 'SC',
    badgeColor: '#6366f1',
    badge: 'Senior SLP'
  },
  {
    id: 'evaluator',
    name: 'Academic Project Evaluator',
    role: 'Dept. of Information Technology (External Reviewer)',
    email: 'evaluator@it.btech.edu',
    initials: 'EV',
    badgeColor: '#10b981',
    badge: 'Reviewer Mode'
  }
];

export default function LoginPage({ onLogin, currentUser, onReturnToDashboard }) {
  const [mode, setMode] = useState('credentials'); // 'credentials' | 'demo'
  const [email, setEmail] = useState('dr.suprathik@acoustiscreen.health');
  const [password, setPassword] = useState('AcoustiScreen2026!');
  const [role, setRole] = useState('Chief Speech-Language Pathologist (SLP)');
  const [showPassword, setShowPassword] = useState(false);
  const [rememberMe, setRememberMe] = useState(true);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [statusMessage, setStatusMessage] = useState(null);

  const handleCredentialsSubmit = (e) => {
    e.preventDefault();
    if (!email.trim()) {
      setStatusMessage({ type: 'error', text: 'Please enter a valid clinical email address.' });
      return;
    }
    if (!password.trim() || password.length < 4) {
      setStatusMessage({ type: 'error', text: 'Password must be at least 4 characters long.' });
      return;
    }

    setIsSubmitting(true);
    setStatusMessage(null);

    // Simulate cryptographic authorization / session establishment
    setTimeout(() => {
      setIsSubmitting(false);
      const user = {
        name: email.split('@')[0].replace(/[._]/g, ' ').replace(/\b\w/g, l => l.toUpperCase()),
        email: email,
        role: role,
        initials: email.substring(0, 2).toUpperCase(),
        rememberMe: rememberMe,
        loginTime: new Date().toISOString()
      };
      onLogin(user);
    }, 450);
  };

  const handleDemoLogin = (profile) => {
    setIsSubmitting(true);
    setTimeout(() => {
      setIsSubmitting(false);
      const user = {
        name: profile.name,
        email: profile.email,
        role: profile.role,
        initials: profile.initials,
        badge: profile.badge,
        loginTime: new Date().toISOString()
      };
      onLogin(user);
    }, 250);
  };

  return (
    <div className="login-viewport">
      {/* Ambient background glow elements */}
      <div className="login-glow-orb top-left" />
      <div className="login-glow-orb bottom-right" />

      <div className="login-card-container">
        {/* ================= LEFT COLUMN: CLINICAL SHOWCASE ================= */}
        <div className="login-hero-pane">
          <div>
            <div className="hero-header-brand">
              <div className="hero-brand-icon">
                <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.3" strokeLinecap="round" strokeLinejoin="round">
                  <path d="M12 2v20M17 5v14M7 8v8M22 10v4M2 10v4" />
                </svg>
              </div>
              <div>
                <h1 className="hero-brand-name">
                  AcoustiScreen <span>AI</span>
                </h1>
                <p className="hero-brand-tagline">
                  Speech Disorder Detection & Clinical Biomarker Suite
                </p>
              </div>
            </div>

            {/* Live Waveform Indicator */}
            <div className="soundwave-visualizer">
              <div className="visualizer-label">
                <span>Multi-Spectral Acoustic Pipeline</span>
                <span className="pulse-badge">40 Biomarkers Active</span>
              </div>
              <div className="wave-bars-row">
                {[45, 78, 30, 92, 60, 85, 40, 95, 70, 50, 80, 65, 90, 35, 75, 55, 88, 48, 72, 60].map((h, i) => (
                  <div
                    key={i}
                    className="wave-bar"
                    style={{
                      height: `${h}%`,
                      animationDelay: `${(i * 0.08).toFixed(2)}s`
                    }}
                  />
                ))}
              </div>
            </div>

            {/* System Capabilities List */}
            <div className="feature-list">
              <div className="feature-item">
                <div className="feature-icon-pill">✓</div>
                <div>
                  <strong>Multi-Class Pathology Classification:</strong> Real-time detection of Dysarthria, Dysphonia, Stuttering, and Normal Speech.
                </div>
              </div>
              <div className="feature-item">
                <div className="feature-icon-pill">✓</div>
                <div>
                  <strong>Acoustic & Prosodic Profiling:</strong> 13 MFCCs, F0 pitch contour, RMS energy, and spectral harmonics.
                </div>
              </div>
              <div className="feature-item">
                <div className="feature-icon-pill">✓</div>
                <div>
                  <strong>Benchmarked ML Ensemble:</strong> SVM (RBF Kernel 95.8%), Random Forest, and calibrated probability matrices.
                </div>
              </div>
            </div>
          </div>

          <div className="hero-footer-note">
            Department of Information Technology • Final Year B.Tech Major Project
            <br />
            Secure MySQL 3306 persistence & HIPAA-conscious preliminary screening protocol.
          </div>
        </div>

        {/* ================= RIGHT COLUMN: LOGIN FORM ================= */}
        <div className="login-form-pane">
          <div className="form-header-box">
            <h2 className="form-title">Clinical Staff Access</h2>
            <p className="form-subtext">
              Sign in to initiate automated acoustic patient screenings and model metrics.
            </p>
          </div>

          {currentUser && onReturnToDashboard && (
            <div style={{
              background: 'rgba(14, 165, 233, 0.12)',
              border: '1px solid rgba(14, 165, 233, 0.3)',
              borderRadius: '10px',
              padding: '10px 14px',
              marginBottom: '16px',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'space-between',
              gap: '10px'
            }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '8px', fontSize: '0.85rem', color: '#e2e8f0' }}>
                <span style={{ width: '8px', height: '8px', borderRadius: '50%', background: '#10b981', display: 'inline-block' }} />
                <span>Signed in as <strong>{currentUser.name}</strong></span>
              </div>
              <button
                type="button"
                onClick={onReturnToDashboard}
                style={{
                  background: '#0284c7',
                  color: '#ffffff',
                  border: 'none',
                  borderRadius: '6px',
                  padding: '6px 12px',
                  fontSize: '0.8rem',
                  fontWeight: '600',
                  cursor: 'pointer',
                  whiteSpace: 'nowrap'
                }}
              >
                Go to Studio &rarr;
              </button>
            </div>
          )}

          {/* Mode Switcher */}
          <div className="login-mode-tabs" role="tablist">
            <button
              id="tab-credentials"
              type="button"
              className={`mode-tab-btn ${mode === 'credentials' ? 'active' : ''}`}
              onClick={() => { setMode('credentials'); setStatusMessage(null); }}
              role="tab"
              aria-selected={mode === 'credentials'}
            >
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2" />
                <circle cx="12" cy="7" r="4" />
              </svg>
              Standard Sign In
            </button>
            <button
              id="tab-demo"
              type="button"
              className={`mode-tab-btn ${mode === 'demo' ? 'active' : ''}`}
              onClick={() => { setMode('demo'); setStatusMessage(null); }}
              role="tab"
              aria-selected={mode === 'demo'}
            >
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2" />
              </svg>
              1-Click Demo Access
            </button>
          </div>

          {/* Status Alert Banner */}
          {statusMessage && (
            <div className={`login-alert ${statusMessage.type}`}>
              <span>{statusMessage.text}</span>
            </div>
          )}

          {/* STANDARD CREDENTIALS FORM */}
          {mode === 'credentials' && (
            <form onSubmit={handleCredentialsSubmit} id="clinical-login-form">
              <div className="form-group">
                <label className="form-label" htmlFor="login-email">Clinical Email</label>
                <div className="input-with-icon">
                  <span className="input-icon">
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                      <rect width="20" height="16" x="2" y="4" rx="2" />
                      <path d="m22 7-8.97 5.7a1.94 1.94 0 0 1-2.06 0L2 7" />
                    </svg>
                  </span>
                  <input
                    id="login-email"
                    type="email"
                    className="form-input"
                    value={email}
                    onChange={(e) => setEmail(e.target.value)}
                    placeholder="doctor.name@hospital.org"
                    required
                  />
                </div>
              </div>

              <div className="form-group">
                <label className="form-label" htmlFor="login-password">Password</label>
                <div className="input-with-icon">
                  <span className="input-icon">
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                      <rect width="18" height="11" x="3" y="11" rx="2" ry="2" />
                      <path d="M7 11V7a5 5 0 0 1 10 0v4" />
                    </svg>
                  </span>
                  <input
                    id="login-password"
                    type={showPassword ? 'text' : 'password'}
                    className="form-input"
                    value={password}
                    onChange={(e) => setPassword(e.target.value)}
                    placeholder="Enter password"
                    required
                  />
                  <button
                    type="button"
                    className="password-toggle-btn"
                    onClick={() => setShowPassword(!showPassword)}
                    aria-label={showPassword ? 'Hide password' : 'Show password'}
                  >
                    {showPassword ? (
                      <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                        <path d="M9.88 9.88a3 3 0 1 0 4.24 4.24" />
                        <path d="M10.73 5.08A10.43 10.43 0 0 1 12 5c7 0 10 7 10 7a13.16 13.16 0 0 1-1.67 2.68" />
                        <path d="M6.61 6.61A13.526 13.526 0 0 0 2 12s3 7 10 7a9.74 9.74 0 0 0 5.39-1.61" />
                        <line x1="2" y1="2" x2="22" y2="22" />
                      </svg>
                    ) : (
                      <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                        <path d="M2 12s3-7 10-7 10 7 10 7-3 7-10 7-10-7-10-7Z" />
                        <circle cx="12" cy="12" r="3" />
                      </svg>
                    )}
                  </button>
                </div>
              </div>

              <div className="form-group">
                <label className="form-label" htmlFor="login-role">Clinical Role</label>
                <div className="input-with-icon">
                  <span className="input-icon">
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                      <path d="M16 21v-2a4 4 0 0 0-4-4H6a4 4 0 0 0-4 4v2" />
                      <circle cx="9" cy="7" r="4" />
                      <path d="M22 21v-2a4 4 0 0 0-3-3.87" />
                      <path d="M16 3.13a4 4 0 0 1 0 7.75" />
                    </svg>
                  </span>
                  <select
                    id="login-role"
                    className="form-input"
                    value={role}
                    onChange={(e) => setRole(e.target.value)}
                  >
                    <option value="Chief Speech-Language Pathologist (SLP)">Chief Speech-Language Pathologist (SLP)</option>
                    <option value="Senior Voice & Neuro-Pathology Examiner">Senior Voice & Neuro-Pathology Examiner</option>
                    <option value="Clinical Audio Research Scientist">Clinical Audio Research Scientist</option>
                    <option value="Academic Project Reviewer">Academic Project Reviewer</option>
                  </select>
                </div>
              </div>

              <div className="form-options-row">
                <label className="remember-label">
                  <input
                    type="checkbox"
                    className="remember-checkbox"
                    checked={rememberMe}
                    onChange={(e) => setRememberMe(e.target.checked)}
                  />
                  <span>Keep session active</span>
                </label>
                <button
                  type="button"
                  className="forgot-link"
                  onClick={() => setStatusMessage({ type: 'info', text: 'Demo access active: You can use any password or switch to 1-Click Demo.' })}
                >
                  Need access?
                </button>
              </div>

              <button
                type="submit"
                id="login-submit-btn"
                className="login-submit-btn"
                disabled={isSubmitting}
              >
                {isSubmitting ? (
                  <>
                    <span className="spinner-border" />
                    <span>Verifying Clinical Authorization...</span>
                  </>
                ) : (
                  <>
                    <span>Enter Screening Studio</span>
                    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                      <path d="M5 12h14M12 5l7 7-7 7" />
                    </svg>
                  </>
                )}
              </button>
            </form>
          )}

          {/* 1-CLICK DEMO PROFILES */}
          {mode === 'demo' && (
            <div>
              <div className="demo-profiles-grid">
                {DEMO_PROFILES.map((prof) => (
                  <button
                    key={prof.id}
                    type="button"
                    className="demo-profile-card"
                    onClick={() => handleDemoLogin(prof)}
                    disabled={isSubmitting}
                  >
                    <div className="profile-avatar" style={{ borderColor: prof.badgeColor, color: prof.badgeColor }}>
                      {prof.initials}
                    </div>
                    <div className="profile-info">
                      <div className="profile-name">{prof.name}</div>
                      <div className="profile-role">{prof.role}</div>
                    </div>
                    <div className="profile-arrow">
                      <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                        <path d="M9 18l6-6-6-6" />
                      </svg>
                    </div>
                  </button>
                ))}
              </div>

              <button
                type="button"
                className="login-submit-btn"
                style={{ background: 'rgba(255,255,255,0.06)', border: '1px solid rgba(255,255,255,0.15)', color: '#cbd5e1' }}
                onClick={() => handleDemoLogin(DEMO_PROFILES[0])}
                disabled={isSubmitting}
              >
                <span>Instant Launch as Lead Researcher</span>
                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <path d="M5 12h14M12 5l7 7-7 7" />
                </svg>
              </button>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
