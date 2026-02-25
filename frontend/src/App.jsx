import { useState, useEffect, useCallback, useRef } from "react";
import {
  getShows,
  deleteShow,
  playShow,
  startLoopback,
  stopPlayback,
  getStatus,
  generateShow,
  getGenerationStatus,
} from "./api";

// ==========================================
// Helper: Format seconds → "3:56"
// ==========================================
function formatDuration(seconds) {
  if (!seconds) return "0:00";
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60);
  return `${m}:${s.toString().padStart(2, "0")}`;
}

// ==========================================
// Helper: RGB array → CSS color string
// ==========================================
function rgbToCSS([r, g, b]) {
  return `rgb(${r}, ${g}, ${b})`;
}

// ==========================================
// Main App Component
// ==========================================
export default function App() {
  // ── State ──
  const [shows, setShows] = useState([]);
  const [isPlaying, setIsPlaying] = useState(false);
  const [url, setUrl] = useState("");
  const [generating, setGenerating] = useState(false);
  const [genMessage, setGenMessage] = useState("");
  const [genProgress, setGenProgress] = useState(0);
  const [toast, setToast] = useState(null);
  const [existsDialog, setExistsDialog] = useState(null); // { url, title, show_id }
  const [genElapsed, setGenElapsed] = useState(0);
  const pollRef = useRef(null);
  const timerRef = useRef(null);

  // ── Show toast notification ──
  const showToast = useCallback((message, duration = 3000) => {
    setToast(message);
    setTimeout(() => setToast(null), duration);
  }, []);

  // ── Load shows from API ──
  const loadShows = useCallback(async () => {
    try {
      const data = await getShows();
      setShows(data);
    } catch (err) {
      console.error("Failed to load shows:", err);
    }
  }, []);

  // ── Check playback status ──
  const checkStatus = useCallback(async () => {
    try {
      const data = await getStatus();
      setIsPlaying(data.playing);
    } catch (err) {
      console.error("Status check failed:", err);
    }
  }, []);

  // ── On mount: load shows + start status polling ──
  useEffect(() => {
    loadShows();
    checkStatus();
    const interval = setInterval(checkStatus, 2000);
    return () => clearInterval(interval);
  }, [loadShows, checkStatus]);

  // ── Handle Generate ──
  const handleGenerate = async (regenerate = false) => {
    const targetUrl = existsDialog?.url || url;
    if (!targetUrl.trim()) return;

    setExistsDialog(null);
    setGenerating(true);
    setGenMessage("Starting...");
    setGenProgress(5);
    setGenElapsed(0);

    // Start elapsed time counter
    timerRef.current = setInterval(() => setGenElapsed((t) => t + 1), 1000);

    try {
      const res = await generateShow(targetUrl.trim(), regenerate);

      // API says "this show already exists" — ask user
      if (res.status === "exists") {
        setGenerating(false);
        setExistsDialog({
          url: targetUrl,
          title: res.title,
          show_id: res.show_id,
        });
        return;
      }

      // Generation started — poll for progress
      pollRef.current = setInterval(async () => {
        try {
          const status = await getGenerationStatus();
          setGenMessage(status.message);
          setGenProgress(status.progress);

          if (!status.active) {
            clearInterval(pollRef.current);
            clearInterval(timerRef.current);
            pollRef.current = null;
            timerRef.current = null;
            setGenerating(false);
            setUrl("");

            if (status.progress === 100) {
              showToast(`✅ Show "${status.show_id}" generated!`);
              loadShows();
            } else {
              showToast(`❌ ${status.message}`);
            }
          }
        } catch (err) {
          clearInterval(pollRef.current);
          clearInterval(timerRef.current);
          setGenerating(false);
        }
      }, 1000);  // Poll every 1s for snappy updates
    } catch (err) {
      setGenerating(false);
      showToast(`❌ ${err.message}`);
    }
  };

  // Cleanup polling on unmount
  useEffect(() => {
    return () => {
      if (pollRef.current) clearInterval(pollRef.current);
      if (timerRef.current) clearInterval(timerRef.current);
    };
  }, []);

  // ── Handle Play (Synced Mode) ──
  const handlePlay = async (showId) => {
    try {
      await playShow(showId);
      setIsPlaying(true);
      showToast("▶ Playing show (synced mode)");
    } catch (err) {
      showToast(`❌ ${err.message}`);
    }
  };

  // ── Handle Loopback Mode ──
  const handleLoopback = async (showId = null) => {
    try {
      await startLoopback(showId);
      setIsPlaying(true);
      showToast("🎧 Loopback mode active — play any audio!");
    } catch (err) {
      showToast(`❌ ${err.message}`);
    }
  };

  // ── Handle Stop ──
  const handleStop = async () => {
    try {
      await stopPlayback();
      setIsPlaying(false);
      showToast("⏹ Playback stopped");
    } catch (err) {
      showToast(`❌ ${err.message}`);
    }
  };

  // ── Handle Delete ──
  const handleDelete = async (showId) => {
    if (!window.confirm(`Delete "${showId}"? This cannot be undone.`)) return;
    try {
      await deleteShow(showId);
      showToast(`🗑 Deleted "${showId}"`);
      loadShows();
    } catch (err) {
      showToast(`❌ ${err.message}`);
    }
  };

  // ── Render ──
  return (
    <div className="app">
      {/* ── Header ── */}
      <header className="header">
        <div className="header-left">
          <span className="header-logo">💡</span>
          <div>
            <div className="header-title">DMX Show Manager</div>
            <div className="header-subtitle">
              AI-powered concert lighting engine
            </div>
          </div>
        </div>

        <div className={`status-badge ${isPlaying ? "playing" : "idle"}`}>
          <span className="status-dot"></span>
          {isPlaying ? "Playing" : "Idle"}
        </div>
      </header>

      {/* ── YouTube Generator Bar ── */}
      <div className="generator-bar">
        <input
          className="generator-input"
          type="text"
          placeholder="Paste YouTube URL to generate a new AI light show..."
          value={url}
          onChange={(e) => setUrl(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && handleGenerate()}
          disabled={generating}
        />
        <button
          className="btn btn-primary"
          onClick={() => handleGenerate()}
          disabled={generating || !url.trim()}
        >
          {generating ? (
            <>
              <span className="spinner"></span> Generating...
            </>
          ) : (
            "⚡ Generate"
          )}
        </button>
      </div>

      {/* ── Generation Progress ── */}
      {generating && (
        <div className="progress-section">
          <div className="progress-label">
            <span className="spinner"></span>
            {genMessage}
            <span style={{ marginLeft: 'auto', color: 'var(--text-muted)', fontSize: 12 }}>
              {Math.floor(genElapsed / 60)}:{(genElapsed % 60).toString().padStart(2, '0')}
            </span>
          </div>
          <div className="progress-track">
            <div
              className="progress-fill"
              style={{ width: `${genProgress}%` }}
            ></div>
          </div>
        </div>
      )}

      {/* ── Global Controls Bar ── */}
      <div className="controls-bar">
        <button
          className="btn btn-secondary"
          onClick={() => handleLoopback()}
          disabled={isPlaying}
        >
          🎧 Live Loopback Mode
        </button>
        {isPlaying && (
          <button className="btn btn-danger" onClick={handleStop}>
            ⏹ Stop
          </button>
        )}
      </div>

      {/* ── Show Library ── */}
      <div className="section-title">🎵 Show Library</div>

      {shows.length === 0 ? (
        <div className="empty-state">
          <div className="empty-state-icon">🎶</div>
          <div className="empty-state-text">No shows yet</div>
          <div className="empty-state-hint">
            Paste a YouTube URL above to generate your first AI light show
          </div>
        </div>
      ) : (
        <div className="shows-grid">
          {shows.map((show) => (
            <div
              className="show-card"
              key={show.show_id}
              style={{
                "--card-color-1": show.palette_preview?.[0]
                  ? rgbToCSS(show.palette_preview[0].color_1)
                  : undefined,
                "--card-color-2": show.palette_preview?.[1]
                  ? rgbToCSS(show.palette_preview[1].color_1)
                  : undefined,
              }}
            >
              {/* Card Header */}
              <div className="show-card-header">
                <div className="show-card-title" title={show.title}>
                  {show.title}
                </div>
                <button
                  className="btn btn-icon btn-danger btn-sm"
                  onClick={() => handleDelete(show.show_id)}
                  title="Delete show"
                >
                  ✕
                </button>
              </div>

              {/* Metadata */}
              <div className="show-card-meta">
                <span>🕐 {formatDuration(show.duration)}</span>
                <span>🎨 {show.num_cues || show.num_phrases} cues</span>
                <span>📅 {show.created_at?.split(" ")[0]}</span>
              </div>

              {/* Color Palette Preview */}
              <div className="palette-preview">
                {show.palette_preview?.map((p, i) => (
                  <div key={i} style={{ display: "flex", gap: 3 }}>
                    <div
                      className="palette-dot"
                      style={{ background: rgbToCSS(p.color_1) }}
                      title={`${p.name} — Color 1`}
                    />
                    <div
                      className="palette-dot"
                      style={{ background: rgbToCSS(p.color_2) }}
                      title={`${p.name} — Color 2`}
                    />
                  </div>
                ))}
              </div>

              {/* Action Buttons */}
              <div className="show-card-actions">
                <button
                  className="btn btn-success btn-sm"
                  onClick={() => handlePlay(show.show_id)}
                  disabled={isPlaying}
                >
                  ▶ Play
                </button>
                <button
                  className="btn btn-secondary btn-sm"
                  onClick={() => handleLoopback(show.show_id)}
                  disabled={isPlaying}
                  title="Use this show's AI palettes in live loopback mode"
                >
                  🎧 Loopback
                </button>
              </div>
            </div>
          ))}
        </div>
      )}

      {/* ── "Show Exists" Dialog ── */}
      {existsDialog && (
        <div className="dialog-overlay" onClick={() => setExistsDialog(null)}>
          <div className="dialog" onClick={(e) => e.stopPropagation()}>
            <h3>Show Already Exists</h3>
            <p>
              <strong>{existsDialog.title}</strong> is already in your library.
              Would you like to use the existing show, or regenerate a fresh AI
              lighting plan?
            </p>
            <div className="dialog-actions">
              <button
                className="btn btn-secondary"
                onClick={() => setExistsDialog(null)}
              >
                Use Existing
              </button>
              <button
                className="btn btn-primary"
                onClick={() => handleGenerate(true)}
              >
                ⚡ Regenerate AI Plan
              </button>
            </div>
          </div>
        </div>
      )}

      {/* ── Toast Notification ── */}
      {toast && <div className="toast">{toast}</div>}
    </div>
  );
}
