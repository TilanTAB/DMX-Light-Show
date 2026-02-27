/**
 * API Service Layer
 * 
 * Every function here maps 1:1 to a FastAPI endpoint in app.py.
 * Components import these functions instead of writing raw fetch() calls.
 * 
 * If the backend URL ever changes, you update ONLY this constant:
 */
const API_BASE = "http://localhost:8000";

/**
 * Helper: wraps fetch() with error handling and JSON parsing.
 * Every API call flows through this single function.
 */
async function apiFetch(path, options = {}) {
    const url = `${API_BASE}${path}`;

    const res = await fetch(url, {
        headers: { "Content-Type": "application/json", ...options.headers },
        ...options,
    });

    // FastAPI returns JSON error bodies — parse them for the UI
    if (!res.ok) {
        const error = await res.json().catch(() => ({ detail: res.statusText }));
        throw new Error(error.detail || `API Error ${res.status}`);
    }

    return res.json();
}

// ==========================================
// SHOW LIBRARY
// ==========================================

/** GET /api/shows → Returns array of saved shows */
export function getShows() {
    return apiFetch("/api/shows");
}

/** DELETE /api/shows/:id → Removes a show from the library */
export function deleteShow(showId) {
    return apiFetch(`/api/shows/${showId}`, { method: "DELETE" });
}

/** POST /api/shows/import-current → Import current_show.json into library */
export function importCurrentShow() {
    return apiFetch("/api/shows/import-current", { method: "POST" });
}

// ==========================================
// PLAYBACK CONTROL
// ==========================================

/** POST /api/play → Start synced WAV playback of a show */
export function playShow(showId) {
    return apiFetch("/api/play", {
        method: "POST",
        body: JSON.stringify({ show_id: showId }),
    });
}

/** POST /api/loopback → Start live WASAPI loopback mode */
export function startLoopback(showId = null, profileId = null) {
    return apiFetch("/api/loopback", {
        method: "POST",
        body: JSON.stringify({ show_id: showId, profile_id: profileId }),
    });
}

/** POST /api/stop → Stop any active playback */
export function stopPlayback() {
    return apiFetch("/api/stop", { method: "POST" });
}

/** GET /api/status → Check if something is currently playing */
export function getStatus() {
    return apiFetch("/api/status");
}

// ==========================================
// SHOW GENERATION
// ==========================================

/** 
 * POST /api/shows/generate → Start AI show generation from YouTube URL 
 * Supports optional time range trimming via start_time/end_time (seconds).
 */
export function generateShow(youtubeUrl, regenerate = false, startTime = null, endTime = null) {
    const body = { url: youtubeUrl, regenerate };
    if (startTime !== null) body.start_time = startTime;
    if (endTime !== null) body.end_time = endTime;
    return apiFetch("/api/shows/generate", {
        method: "POST",
        body: JSON.stringify(body),
    });
}

/** GET /api/shows/generate/status → Poll generation progress */
export function getGenerationStatus() {
    return apiFetch("/api/shows/generate/status");
}

// ==========================================
// PLAYBACK CONTROL
// ==========================================

/** GET /api/playback/position → Get current playback position, duration, state */
export function getPlaybackPosition() {
    return apiFetch("/api/playback/position");
}

/** POST /api/playback/seek → Seek to a position (seconds) */
export function seekPlayback(position) {
    return apiFetch("/api/playback/seek", {
        method: "POST",
        body: JSON.stringify({ position }),
    });
}

/** POST /api/playback/pause → Pause synced playback */
export function pausePlayback() {
    return apiFetch("/api/playback/pause", { method: "POST" });
}

/** POST /api/playback/resume → Resume paused playback */
export function resumePlayback() {
    return apiFetch("/api/playback/resume", { method: "POST" });
}

// ==========================================
// PROFILES
// ==========================================

/** GET /api/profiles → List all profiles */
export function getProfiles() {
    return apiFetch("/api/profiles");
}

/** GET /api/profiles/:id → Get profile details */
export function getProfile(profileId) {
    return apiFetch(`/api/profiles/${profileId}`);
}

/** POST /api/profiles/:id/activate → Activate profile + restart loopback */
export function activateProfile(profileId) {
    return apiFetch(`/api/profiles/${profileId}/activate`, { method: "POST" });
}
