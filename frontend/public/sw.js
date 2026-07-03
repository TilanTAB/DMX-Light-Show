// DMX Light Show — Service Worker (minimal)
// This enables PWA installability and caches the app shell for instant loads.
// Since this app depends on a live backend (FastAPI), we use a network-first
// strategy: always try the network, fall back to cache only for static assets.

const CACHE_NAME = 'dmx-show-v1';
const APP_SHELL = [
  '/',
  '/index.html',
  '/manifest.json',
  '/icon-512.png',
];

// On install: pre-cache the app shell (HTML, manifest, icon)
self.addEventListener('install', (event) => {
  event.waitUntil(
    caches.open(CACHE_NAME).then((cache) => cache.addAll(APP_SHELL))
  );
  self.skipWaiting(); // Activate immediately
});

// On activate: clean up old caches
self.addEventListener('activate', (event) => {
  event.waitUntil(
    caches.keys().then((names) =>
      Promise.all(
        names
          .filter((name) => name !== CACHE_NAME)
          .map((name) => caches.delete(name))
      )
    )
  );
  self.clients.claim(); // Take control of all pages immediately
});

// Fetch strategy:
// - API calls (/api/*): ALWAYS network-only (never cache API responses)
// - Static assets: network-first, fall back to cache
self.addEventListener('fetch', (event) => {
  const url = new URL(event.request.url);

  // Never cache API calls — they must always hit the live backend
  if (url.pathname.startsWith('/api')) {
    return; // Let the browser handle it normally (network only)
  }

  event.respondWith(
    fetch(event.request)
      .then((response) => {
        // Cache successful responses for static assets
        if (response.ok) {
          const clone = response.clone();
          caches.open(CACHE_NAME).then((cache) => cache.put(event.request, clone));
        }
        return response;
      })
      .catch(() => caches.match(event.request)) // Offline fallback
  );
});
