/**
 * WebCameraView.jsx
 * -------------------
 * Camera/hand-tracking/prediction pipeline, now with HYBRID static+motion
 * support, mirroring asl_recognizer_hybrid.py's routing:
 *
 *   hand roughly still  -> normalized single-frame landmarks -> {"landmarks": [...]}
 *   hand moving          -> rolling window of RAW frames      -> {"sequence": [[...], ...]}
 *
 * Motion detection (wrist speed over recent frames) happens HERE, client-side
 * — mirrors HybridRecognizer._is_moving(). The backend/ML server stay fully
 * stateless; this component is where the "does this look like J/Z" gating
 * decision is made, same as smoothing/sentence-building already were.
 *
 * On mount, fetches /model_info from the backend to learn whether motion
 * prediction is available and what window size the trained seq model
 * expects — rather than hardcoding a guess that could silently mismatch.
 */

import React, { useEffect, useRef, useState } from "react";
import { normalizeLandmarks } from "./lib/normalize";
import { extractRawLandmarks } from "./lib/raw_landmarks";

const BACKEND_WS_URL = "ws://localhost:8000/predict";
const BACKEND_HTTP_URL = "http://localhost:8000";
const SMOOTHING_WINDOW = 10;
const CONFIDENCE_THRESH = 0.80;
const STABLE_THRESH = 20;

// Motion gate — same meaning as MOTION_SPEED_THRESH / MOTION_HISTORY in
// asl_recognizer_hybrid.py. Tune against real footage; read the live speed
// value (logged to console when DEBUG_MOTION is true) while signing J/Z.
const MOTION_SPEED_THRESH = 0.005;
const MOTION_HISTORY = 15; // frames averaged for the speed estimate
const DEBUG_MOTION = false;

const PALM_CONNECTIONS = [[0, 1], [0, 5], [9, 13], [13, 17], [5, 9], [0, 17]];
const THUMB_CONNECTIONS = [[1, 2], [2, 3], [3, 4]];
const INDEX_CONNECTIONS = [[5, 6], [6, 7], [7, 8]];
const MIDDLE_CONNECTIONS = [[9, 10], [10, 11], [11, 12]];
const RING_CONNECTIONS = [[13, 14], [14, 15], [15, 16]];
const PINKY_CONNECTIONS = [[17, 18], [18, 19], [19, 20]];
const HAND_CONNECTIONS = [
  ...PALM_CONNECTIONS, ...THUMB_CONNECTIONS, ...INDEX_CONNECTIONS,
  ...MIDDLE_CONNECTIONS, ...RING_CONNECTIONS, ...PINKY_CONNECTIONS,
];

export default function WebCameraView() {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const wsRef = useRef(null);
  const wsReadyRef = useRef(false);

  const predBufferRef = useRef([]);
  const confBufferRef = useRef([]);
  const lastAppendedRef = useRef("");
  const stableCountRef = useRef(0);

  // ── Motion-gate state ──────────────────────────────────────────────────
  const wristHistRef = useRef([]);  // recent raw wrist [x,y] pairs
  const rawWindowRef = useRef([]);  // rolling buffer of raw frames for the seq model
  const motionInfoRef = useRef({ available: false, window: 30 }); // from /model_info
  const modeRef = useRef("STATIC"); // "STATIC" | "MOTION" | "MOTION*" (moving, no seq model/buffer yet)

  const [status, setStatus] = useState("Loading hand-tracking model…");
  const [letter, setLetter] = useState("");
  const [confidence, setConfidence] = useState(null);
  const [sentence, setSentence] = useState("");
  const [mode, setMode] = useState("STATIC");

  const handleClear = () => {
    setSentence("");
    lastAppendedRef.current = "";
    stableCountRef.current = 0;
  };

  // ── Smoothing + sentence-building — unified across static/motion, since
  //    both ultimately just produce a letter string, same as
  //    HybridRecognizer._finalize() unifies them server/recognizer-side ──
  const handlePrediction = (pLetter, pConfidence) => {
    predBufferRef.current.push(pLetter);
    confBufferRef.current.push(pConfidence);
    if (predBufferRef.current.length > SMOOTHING_WINDOW) {
      predBufferRef.current.shift();
      confBufferRef.current.shift();
    }

    let smoothLetter, smoothConf;
    if (predBufferRef.current.length === SMOOTHING_WINDOW) {
      const counts = {};
      for (const l of predBufferRef.current) counts[l] = (counts[l] || 0) + 1;
      smoothLetter = Object.keys(counts).reduce((a, b) => (counts[a] > counts[b] ? a : b));
      const matching = confBufferRef.current.filter((_, i) => predBufferRef.current[i] === smoothLetter);
      smoothConf = matching.reduce((a, b) => a + b, 0) / matching.length;
    } else {
      smoothLetter = pLetter;
      smoothConf = pConfidence;
    }

    if (!smoothLetter || smoothConf < CONFIDENCE_THRESH) {
      setLetter("");
      setConfidence(smoothConf);
      return;
    }

    if (smoothLetter === lastAppendedRef.current) {
      stableCountRef.current++;
    } else {
      stableCountRef.current = 0;
      lastAppendedRef.current = smoothLetter;
    }
    if (stableCountRef.current === STABLE_THRESH) {
      setSentence((prev) => prev + smoothLetter);
    }

    setLetter(smoothLetter);
    setConfidence(smoothConf);
  };

  const drawLandmarks = (ctx, canvas, landmarks) => {
    ctx.strokeStyle = "#3b82f6";
    ctx.lineWidth = 2;
    for (const [a, b] of HAND_CONNECTIONS) {
      const pa = landmarks[a];
      const pb = landmarks[b];
      ctx.beginPath();
      ctx.moveTo(pa.x * canvas.width, pa.y * canvas.height);
      ctx.lineTo(pb.x * canvas.width, pb.y * canvas.height);
      ctx.stroke();
    }
    ctx.fillStyle = "#4ade80";
    for (const p of landmarks) {
      ctx.beginPath();
      ctx.arc(p.x * canvas.width, p.y * canvas.height, 3, 0, 2 * Math.PI);
      ctx.fill();
    }
  };

  // ── Motion gate — port of HybridRecognizer._is_moving() ─────────────────
  const isMoving = (rawFlat) => {
    // landmark 0 (wrist) is indices [0,1,2]; only need x,y for the gate.
    const wristXY = [rawFlat[0], rawFlat[1]];
    wristHistRef.current.push(wristXY);
    if (wristHistRef.current.length > MOTION_HISTORY) wristHistRef.current.shift();
    if (wristHistRef.current.length < 2) return { moving: false, speed: 0 };

    let total = 0;
    for (let i = 1; i < wristHistRef.current.length; i++) {
      const dx = wristHistRef.current[i][0] - wristHistRef.current[i - 1][0];
      const dy = wristHistRef.current[i][1] - wristHistRef.current[i - 1][1];
      total += Math.sqrt(dx * dx + dy * dy);
    }
    const speed = total / (wristHistRef.current.length - 1);
    if (DEBUG_MOTION) console.log("wrist speed:", speed.toFixed(5));
    return { moving: speed > MOTION_SPEED_THRESH, speed };
  };

  const connectWebSocket = () => {
    const ws = new WebSocket(BACKEND_WS_URL);
    wsRef.current = ws;
    ws.onopen = () => {
      wsReadyRef.current = true;
      setStatus("Connected");
    };
    ws.onclose = () => {
      wsReadyRef.current = false;
      setStatus("Disconnected — retrying…");
      setTimeout(connectWebSocket, 2000);
    };
    ws.onerror = () => ws.close();
    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
      if (data.error) return;
      handlePrediction(data.letter, data.confidence);
    };
  };

  useEffect(() => {
    // Ask the backend (which relays to the ML server) whether motion
    // prediction is available and what window size to buffer — avoids
    // hardcoding a guess that could silently mismatch the trained model.
    fetch(`${BACKEND_HTTP_URL}/model_info`)
      .then((r) => r.json())
      .then((info) => {
        motionInfoRef.current = {
          available: !!info.motion_available,
          window: info.window || 30,
        };
        console.log("[model_info]", motionInfoRef.current);
      })
      .catch((err) => console.warn("Could not fetch /model_info — motion disabled.", err));
  }, []);

  useEffect(() => {
    let handLandmarker;
    let rafId;
    let cancelled = false;

    async function setup() {
      const { FilesetResolver, HandLandmarker } = await import(
        /* @vite-ignore */ "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/+esm"
      );

      const vision = await FilesetResolver.forVisionTasks(
        "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm"
      );
      handLandmarker = await HandLandmarker.createFromOptions(vision, {
        baseOptions: {
          modelAssetPath:
            "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
        },
        runningMode: "VIDEO",
        numHands: 1,
      });
      if (cancelled) return;

      setStatus("Requesting camera…");
      const stream = await navigator.mediaDevices.getUserMedia({ video: true });
      if (cancelled) return;

      const video = videoRef.current;
      video.srcObject = stream;
      await video.play();

      const canvas = canvasRef.current;
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      const ctx = canvas.getContext("2d");

      connectWebSocket();
      setStatus("Running");

      let lastVideoTime = -1;
      function detectLoop() {
        if (cancelled) return;
        if (video.currentTime !== lastVideoTime) {
          lastVideoTime = video.currentTime;

          ctx.save();
          ctx.scale(-1, 1);
          ctx.drawImage(video, -canvas.width, 0, canvas.width, canvas.height);
          ctx.restore();

          const result = handLandmarker.detectForVideo(canvas, performance.now());

          if (result.landmarks && result.landmarks.length > 0) {
            const landmarks = result.landmarks[0];
            drawLandmarks(ctx, canvas, landmarks);

            const raw = extractRawLandmarks(landmarks);
            const { moving } = isMoving(raw);

            const windowSize = motionInfoRef.current.window;
            rawWindowRef.current.push(raw);
            if (rawWindowRef.current.length > windowSize) rawWindowRef.current.shift();

            const bufferReady = rawWindowRef.current.length === windowSize;
            const canUseMotion = motionInfoRef.current.available && moving && bufferReady;

            if (canUseMotion) {
              modeRef.current = "MOTION";
              if (wsReadyRef.current) {
                wsRef.current.send(JSON.stringify({ sequence: rawWindowRef.current }));
              }
            } else {
              modeRef.current = moving ? "MOTION*" : "STATIC"; // * = moving but no seq model/buffer yet
              const normalized = normalizeLandmarks(landmarks);
              if (normalized && wsReadyRef.current) {
                wsRef.current.send(JSON.stringify({ landmarks: normalized }));
              }
            }
            setMode(modeRef.current);
          } else {
            predBufferRef.current = [];
            confBufferRef.current = [];
            wristHistRef.current = [];
            rawWindowRef.current = [];
          }
        }
        rafId = requestAnimationFrame(detectLoop);
      }
      detectLoop();
    }

    setup().catch((err) => {
      console.error(err);
      setStatus(`Error: ${err.message}`);
    });

    return () => {
      cancelled = true;
      if (rafId) cancelAnimationFrame(rafId);
      if (wsRef.current) wsRef.current.close();
    };
  }, []);

  return (
    <div style={styles.page}>
      <h1 style={styles.h1}>Sign Buddy</h1>
      <div style={styles.status}>
        {status} {mode ? <span style={styles.modeTag}>[{mode}]</span> : null}
      </div>

      <div style={styles.videoContainer}>
        <video ref={videoRef} autoPlay playsInline muted style={styles.hiddenVideo} />
        <canvas ref={canvasRef} style={styles.canvas} />
      </div>

      <div style={styles.predictionBar}>
        <span>{letter || "—"}</span>
        {confidence != null && (
          <span style={styles.confidence}>{Math.round(confidence * 100)}%</span>
        )}
      </div>

      <div style={styles.sentenceBox}>
        <label>Spelled so far:</label>
        <div style={styles.sentence}>{sentence}</div>
        <button onClick={handleClear}>Clear</button>
      </div>
    </div>
  );
}

const styles = {
  page: { fontFamily: "system-ui, sans-serif", background: "#111", color: "#eee", textAlign: "center", padding: 20, minHeight: "100vh", width: "100%", boxSizing: "border-box" },
  h1: { marginBottom: 4 },
  status: { marginBottom: 12, fontSize: 14, color: "#999" },
  modeTag: { color: "#80c8e6", marginLeft: 6, fontWeight: "bold" },
  videoContainer: { position: "relative", display: "inline-block" },
  hiddenVideo: { position: "absolute", visibility: "hidden", width: 1, height: 1 },
  canvas: { width: 640, height: 480, borderRadius: 8 },
  predictionBar: { marginTop: 16, fontSize: 48, fontWeight: "bold" },
  confidence: { fontSize: 18, color: "#7ce67c", marginLeft: 12 },
  sentenceBox: { marginTop: 20 },
  sentence: { fontSize: 24, minHeight: 40, background: "#222", borderRadius: 6, padding: 10, margin: "8px auto", maxWidth: 640 },
};
