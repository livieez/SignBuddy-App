/**
 * lib/normalize.js
 * ------------------
 * Same normalization logic as before (wrist-center, then scale by
 * wrist-to-middle-knuckle distance) — matches HandDetector.extract_landmark_array()
 * in hand_tracking_module.py. Now a plain importable module (no <script type=module>
 * tag needed) since it's bundled by Metro like any other project file.
 */

const WRIST_INDEX = 0;
const MIDDLE_MCP_INDEX = 9;

export function normalizeLandmarks(landmarks) {
  if (!landmarks || landmarks.length !== 21) return null;

  const wrist = landmarks[WRIST_INDEX];
  const centered = landmarks.map((p) => ({
    x: p.x - wrist.x,
    y: p.y - wrist.y,
    z: p.z - wrist.z,
  }));

  const ref = centered[MIDDLE_MCP_INDEX];
  const scale = Math.sqrt(ref.x * ref.x + ref.y * ref.y + ref.z * ref.z);

  const scaled = scale > 0
    ? centered.map((p) => ({ x: p.x / scale, y: p.y / scale, z: p.z / scale }))
    : centered;

  const flat = [];
  for (const p of scaled) flat.push(p.x, p.y, p.z);
  return flat;
}
