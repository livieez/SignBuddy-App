/**
 * raw_landmarks.js
 * ------------------
 * For the MOTION path only. Unlike normalize.js (wrist-centered, scaled —
 * used for static letters), this returns landmarks completely UNPROCESSED.
 *
 * Why: motion detection tracks the wrist moving ACROSS frames. If each frame
 * were independently wrist-centered (like normalize.js does), the wrist would
 * sit at (0,0,0) in every single frame — the exact motion signal we're trying
 * to detect would be normalized away. Matches
 * hand_tracking_module.extract_raw_landmark_array on the Python side, and
 * asl_recognizer_hybrid.py's motion path, which explicitly uses raw,
 * un-centered landmarks for this same reason.
 */

export function extractRawLandmarks(landmarks) {
  if (!landmarks || landmarks.length !== 21) return null;
  const flat = [];
  for (const p of landmarks) flat.push(p.x, p.y, p.z);
  return flat; // (63,) — no centering, no scaling
}
