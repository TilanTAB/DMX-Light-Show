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
  getPlaybackPosition,
  seekPlayback,
  pausePlayback,
  resumePlayback,
} from "./api";

// ==========================================
// Helper: Format seconds → "3:56"
// ==========================================
function formatDuration(seconds) {
  if (!seconds && seconds !== 0) return "0:00";
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60);
  return `${m}:${s.toString().padStart(2, "0")}`;
}

// ==========================================
// Helper: Parse "M:SS" or "MM:SS" to seconds
// ==========================================
function parseTimeInput(str) {
  if (!str || !str.trim()) return null;
  const cleaned = str.trim();
  // Accept raw seconds
  if (/^\d+(\.\d+)?$/.test(cleaned)) return parseFloat(cleaned);
  // Accept M:SS or MM:SS
  const parts = cleaned.split(":");
  if (parts.length === 2) {
    const m = parseInt(parts[0], 10);
    const s = parseInt(parts[1], 10);
    if (!isNaN(m) && !isNaN(s)) return m * 60 + s;
  }
  return null;
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
  const [startTime, setStartTime] = useState("");
  const [endTime, setEndTime] = useState("");
  const [generating, setGenerating] = useState(false);
  const [genMessage, setGenMessage] = useState("");
  const [genProgress, setGenProgress] = useState(0);
  const [toast, setToast] = useState(null);
  const [existsDialog, setExistsDialog] = useState(null);
  const [genElapsed, setGenElapsed] = useState(0);

  // Playback position state
  const [playbackPos, setPlaybackPos] = useState(0);
  const [playbackDuration, setPlaybackDuration] = useState(0);
  const [playbackState, setPlaybackState] = useState("stopped");
  const [cueName, setCueName] = useState("");
  const [behavior, setBehavior] = useState("");
  const [isSeeking, setIsSeeking] = useState(false);

  const pollRef = useRef(null);
  const timerRef = useRef(null);
  const toastTimer = useRef(null);
  const positionPollRef = useRef(null);

  // ── Show toast notification (with timer cleanup) ──
  const showToast = useCallback((message, duration = 3000) => {
    clearTimeout(toastTimer.current);
    setToast(message);
    toastTimer.current = setTimeout(() => setToast(null), duration);
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

      // If playing, update position from status endpoint
      if (data.playing && data.playback && !isSeeking) {
        setPlaybackPos(data.playback.position || 0);
        setPlaybackDuration(data.playback.duration || 0);
        setPlaybackState(data.playback.state || "playing");
        setCueName(data.playback.cue_name || "");
        setBehavior(data.playback.behavior || "");
      } else if (!data.playing) {
        setPlaybackState("stopped");
      }
    } catch (err) {
      console.error("Status check failed:", err);
    }
  }, [isSeeking]);

  // ── Position polling (faster when playing) ──
  // FIX #3: Use recursive setTimeout instead of setInterval to prevent 
  // promise accumulation when the backend is slow.
  useEffect(() => {
    if (!isPlaying) return;
    let cancelled = false;

    async function pollPosition() {
      if (cancelled || isSeeking) {
        if (!cancelled) setTimeout(pollPosition, 300);
        return;
      }
      try {
        const data = await getPlaybackPosition();
        if (!cancelled) {
          setPlaybackPos(data.position || 0);
          setPlaybackDuration(data.duration || 0);
          setPlaybackState(data.state || "stopped");
          setCueName(data.cue_name || "");
          setBehavior(data.behavior || "");
        }
      } catch (err) {
        // Ignore errors during polling
      }
      if (!cancelled) setTimeout(pollPosition, 300);
    }

    pollPosition();
    return () => { cancelled = true; };
  }, [isPlaying, isSeeking]);

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

    timerRef.current = setInterval(() => setGenElapsed((t) => t + 1), 1000);

    // Parse optional time range
    const startSec = parseTimeInput(startTime);
    const endSec = parseTimeInput(endTime);

    try {
      const res = await generateShow(targetUrl.trim(), regenerate, startSec, endSec);

      if (res.status === "exists") {
        setGenerating(false);
        setExistsDialog({
          url: targetUrl,
          title: res.title,
          show_id: res.show_id,
        });
        return;
      }

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
            setStartTime("");
            setEndTime("");

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
      }, 1000);
    } catch (err) {
      setGenerating(false);
      showToast(`❌ ${err.message}`);
    }
  };

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
      setPlaybackState("playing");
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
      setPlaybackState("stopped");
      setPlaybackPos(0);
      showToast("⏹ Playback stopped");
    } catch (err) {
      showToast(`❌ ${err.message}`);
    }
  };

  // ── Handle Pause / Resume ──
  const handlePauseResume = async () => {
    try {
      if (playbackState === "paused") {
        await resumePlayback();
        setPlaybackState("playing");
      } else {
        await pausePlayback();
        setPlaybackState("paused");
      }
    } catch (err) {
      showToast(`❌ ${err.message}`);
    }
  };

  // ── Handle Seek ──
  const handleSeek = async (e) => {
    const newPos = parseFloat(e.target.value);
    setPlaybackPos(newPos);
    setIsSeeking(false);
    try {
      await seekPlayback(newPos);
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

      {/* ── Time Range Inputs (optional) ── */}
      <div className="time-range-bar">
        <span className="time-range-label">✂️ Trim (optional):</span>
        <input
          className="time-input"
          type="text"
          placeholder="Start (e.g. 1:30)"
          value={startTime}
          onChange={(e) => setStartTime(e.target.value)}
          disabled={generating}
        />
        <span className="time-range-sep">→</span>
        <input
          className="time-input"
          type="text"
          placeholder="End (e.g. 5:00)"
          value={endTime}
          onChange={(e) => setEndTime(e.target.value)}
          disabled={generating}
        />
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

      {/* ── Now Playing / Player Controls ── */}
      {isPlaying && (
        <div className="player-panel">
          <div className="player-header">
            <span className="player-title">🎵 Now Playing</span>
            <div className="player-meta">
              {cueName && <span className="player-cue">{cueName}</span>}
              {behavior && <span className="player-behavior">{behavior}</span>}
            </div>
          </div>

          <div className="player-slider-row">
            <span className="player-time">{formatDuration(playbackPos)}</span>
            <input
              type="range"
              className="player-slider"
              min="0"
              max={playbackDuration || 1}
              step="0.5"
              value={isSeeking ? undefined : playbackPos}
              onMouseDown={() => setIsSeeking(true)}
              onTouchStart={() => setIsSeeking(true)}
              onChange={(e) => setPlaybackPos(parseFloat(e.target.value))}
              onMouseUp={handleSeek}
              onTouchEnd={handleSeek}
            />
            <span className="player-time">{formatDuration(playbackDuration)}</span>
          </div>

          <div className="player-controls">
            <button className="btn btn-secondary btn-sm" onClick={handlePauseResume}>
              {playbackState === "paused" ? "▶ Resume" : "⏸ Pause"}
            </button>
            <button className="btn btn-danger btn-sm" onClick={handleStop}>
              ⏹ Stop
            </button>
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
        {isPlaying && !isPlaying && (
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

              <div className="show-card-meta">
                <span>🕐 {formatDuration(show.duration)}</span>
                <span>🎨 {show.num_cues || show.num_phrases} cues</span>
                <span>📅 {show.created_at?.split(" ")[0]}</span>
              </div>

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
