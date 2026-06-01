/* Service worker: shell + imágenes locales (y iconos PWA si existen). */
const CACHE_NAME = "abrasito-v2";

const SHELL_ASSETS = [
  "./",
  "./index.html",
  "./manifest.webmanifest",
  "./sw.js",
];

const IMAGE_ASSETS = [
  "./Yo.jpg",
  "./Key.jpg",
  "./tabboard-corcho.jpg",
  "./Florkyflor.png",
];

const ICON_ASSETS = [
  "/AppPlasa/android-chrome-192x192.png",
  "/AppPlasa/android-chrome-512x512.png",
  "/AppPlasa/apple-touch-icon.png",
  "/AppPlasa/favicon-32x32.png",
];

function isImageRequest(url) {
  return /\.(png|jpe?g|webp|gif|svg|ico)(\?|$)/i.test(url.pathname);
}

async function cacheAddAll(cache, urls) {
  await Promise.all(
    urls.map(async (url) => {
      try {
        const res = await fetch(url, { cache: "reload" });
        if (res.ok) await cache.put(url, res);
      } catch (_) {
        /* Algunos iconos /AppPlasa/ pueden no existir en local. */
      }
    })
  );
}

self.addEventListener("install", (event) => {
  event.waitUntil(
    (async () => {
      const cache = await caches.open(CACHE_NAME);
      await cacheAddAll(cache, [...SHELL_ASSETS, ...IMAGE_ASSETS, ...ICON_ASSETS]);
      self.skipWaiting();
    })()
  );
});

self.addEventListener("activate", (event) => {
  event.waitUntil(
    (async () => {
      const keys = await caches.keys();
      await Promise.all(
        keys.filter((k) => k !== CACHE_NAME).map((k) => caches.delete(k))
      );
      await self.clients.claim();
    })()
  );
});

self.addEventListener("message", (event) => {
  if (event.data?.type === "PRECACHE_IMAGES") {
    const urls = event.data.urls || IMAGE_ASSETS;
    event.waitUntil(
      (async () => {
        const cache = await caches.open(CACHE_NAME);
        await cacheAddAll(cache, urls);
      })()
    );
  }
});

self.addEventListener("fetch", (event) => {
  const req = event.request;
  if (req.method !== "GET") return;

  const url = new URL(req.url);
  if (url.origin !== self.location.origin) return;

  const path = url.pathname;
  const isNav =
    req.mode === "navigate" ||
    path.endsWith("/") ||
    path.endsWith("/index.html");

  if (isNav) {
    event.respondWith(
      (async () => {
        try {
          const fresh = await fetch(req);
          const cache = await caches.open(CACHE_NAME);
          cache.put("./index.html", fresh.clone());
          return fresh;
        } catch {
          return (
            (await caches.match("./index.html")) ||
            (await caches.match("./")) ||
            new Response("Sin conexión", { status: 503 })
          );
        }
      })()
    );
    return;
  }

  if (isImageRequest(url) || IMAGE_ASSETS.some((a) => path.endsWith(a.slice(1)))) {
    event.respondWith(
      (async () => {
        const cached = await caches.match(req);
        if (cached) return cached;
        try {
          const res = await fetch(req);
          if (res.ok) {
            const cache = await caches.open(CACHE_NAME);
            cache.put(req, res.clone());
          }
          return res;
        } catch {
          return cached || new Response("", { status: 504 });
        }
      })()
    );
    return;
  }

  event.respondWith(
    (async () => {
      const cached = await caches.match(req);
      if (cached) return cached;
      try {
        const res = await fetch(req);
        if (res.ok && (path.includes(".") || path.endsWith("manifest.webmanifest"))) {
          const cache = await caches.open(CACHE_NAME);
          cache.put(req, res.clone());
        }
        return res;
      } catch {
        return cached || new Response("", { status: 504 });
      }
    })()
  );
});
