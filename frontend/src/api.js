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
export function startLoopback(showId = null) {
    return apiFetch("/api/loopback", {
        method: "POST",
        body: JSON.stringify({ show_id: showId }),
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
 * 
 * If regenerate=false and the show already exists, API returns { status: "exists" }
 * so the UI can ask the user what to do. If regenerate=true, it skips the download
 * and re-runs only the AI plan generation (saves ~2 minutes).
 */
export function generateShow(youtubeUrl, regenerate = false) {
    return apiFetch("/api/shows/generate", {
        method: "POST",
        body: JSON.stringify({ url: youtubeUrl, regenerate }),
    });
}

/** GET /api/shows/generate/status → Poll generation progress */
export function getGenerationStatus() {
    return apiFetch("/api/shows/generate/status");
}
