/** Live MJPEG video feed from the engine. */
export function VideoFeed() {
  return (
    <div className="relative bg-black rounded-xl overflow-hidden flex items-center justify-center min-h-[300px]">
      <img
        src="/api/feed"
        alt="Live video feed"
        className="w-full h-full object-contain"
      />
      {/* Fallback when no feed */}
      <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
        <noscript>
          <span className="text-vertex-muted text-sm">
            Waiting for video feed...
          </span>
        </noscript>
      </div>
    </div>
  );
}
