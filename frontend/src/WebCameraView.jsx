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

// const BACKEND_WS_URL = "ws://localhost:8000/predict";
// const BACKEND_HTTP_URL = "http://localhost:8000";
const BACKEND_WS_URL = "wss://ripcord-imaginary-abacus.ngrok-free.dev/predict"; // using ngrok temporarily to host backend
const BACKEND_HTTP_URL = "https://ripcord-imaginary-abacus.ngrok-free.dev";
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
    ctx.strokeStyle = "#5B9DFF";
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
    <div className="sb-page">
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=Manrope:wght@500;700;800&family=JetBrains+Mono:wght@500;600;700&display=swap');

        :root {
          --bg: #0B0D10;
          --panel: #14171B;
          --panel-border: #262B31;
          --text-primary: #EDEFF2;
          --text-secondary: #8A93A0;
          --accent: #6C8CFF;
          --accent-warm: #F2B84B;
        }
        * { box-sizing: border-box; }

        .sb-page {
          min-height: 100vh;
          width: 100%;
          background: var(--bg);
          color: var(--text-primary);
          font-family: 'Manrope', system-ui, sans-serif;
          display: flex;
          flex-direction: column;
          align-items: center;
          padding: clamp(20px, 5vw, 44px) 20px 60px;
        }

        .sb-header {
          width: 100%;
          max-width: 720px;
          display: flex;
          flex-wrap: wrap;
          align-items: baseline;
          justify-content: space-between;
          gap: 8px;
          margin-bottom: 24px;
        }
        .sb-title {
          font-weight: 800;
          font-size: clamp(22px, 4vw, 28px);
          letter-spacing: -0.02em;
          margin: 0;
        }
        .sb-status {
          display: flex;
          align-items: center;
          gap: 8px;
          font-size: 13px;
          color: var(--text-secondary);
        }
        .sb-status-dot {
          width: 7px; height: 7px;
          border-radius: 50%;
          background: var(--text-secondary);
          transition: background 0.2s ease, box-shadow 0.2s ease;
        }
        .sb-status-dot.is-live {
          background: var(--accent-warm);
          box-shadow: 0 0 0 3px rgba(242,184,75,0.16);
        }
        .sb-mode {
          font-family: 'JetBrains Mono', monospace;
          font-size: 11px;
          font-weight: 600;
          letter-spacing: 0.06em;
          color: var(--accent);
          border: 1px solid rgba(108,140,255,0.35);
          padding: 2px 8px;
          border-radius: 4px;
        }

        .sb-video-frame {
          position: relative;
          width: 100%;
          max-width: 720px;
          aspect-ratio: 4 / 3;
          border-radius: 10px;
          overflow: hidden;
          background: #000;
          border: 1px solid var(--panel-border);
        }
        .sb-video-frame canvas {
          width: 100%;
          height: 100%;
          display: block;
        }
        .sb-corner {
          position: absolute;
          width: 20px; height: 20px;
          border: 2px solid var(--accent);
          opacity: 0.85;
          z-index: 2;
          pointer-events: none;
        }
        .sb-corner--tl { top: 10px; left: 10px; border-right: none; border-bottom: none; }
        .sb-corner--tr { top: 10px; right: 10px; border-left: none; border-bottom: none; }
        .sb-corner--bl { bottom: 10px; left: 10px; border-right: none; border-top: none; }
        .sb-corner--br { bottom: 10px; right: 10px; border-left: none; border-top: none; }

        .sb-readout {
          width: 100%;
          max-width: 720px;
          display: flex;
          flex-wrap: wrap;
          align-items: center;
          gap: 20px;
          margin-top: 20px;
          padding: 18px 20px;
          background: var(--panel);
          border: 1px solid var(--panel-border);
          border-radius: 10px;
        }
        .sb-letter-tile {
          width: 60px; height: 60px;
          flex-shrink: 0;
          display: flex;
          align-items: center;
          justify-content: center;
          font-family: 'JetBrains Mono', monospace;
          font-size: 28px;
          font-weight: 700;
          background: #0F1216;
          border: 1px solid var(--panel-border);
          border-radius: 8px;
          box-shadow: inset 0 1px 0 rgba(255,255,255,0.03);
        }
        .sb-meter { flex: 1; min-width: 160px; }
        .sb-meter-label {
          display: flex;
          justify-content: space-between;
          font-family: 'JetBrains Mono', monospace;
          font-size: 12px;
          color: var(--text-secondary);
          margin-bottom: 6px;
        }
        .sb-meter-value { color: var(--accent-warm); font-weight: 600; }
        .sb-meter-track {
          height: 6px;
          border-radius: 3px;
          background: #0F1216;
          border: 1px solid var(--panel-border);
          overflow: hidden;
        }
        .sb-meter-fill {
          height: 100%;
          background: linear-gradient(90deg, var(--accent), var(--accent-warm));
          transition: width 0.15s ease;
        }

        .sb-sentence-panel { width: 100%; max-width: 720px; margin-top: 20px; }
        .sb-sentence-label {
          font-size: 12px;
          text-transform: uppercase;
          letter-spacing: 0.08em;
          color: var(--text-secondary);
          margin-bottom: 8px;
        }
        .sb-sentence-output {
          min-height: 48px;
          background: var(--panel);
          border: 1px solid var(--panel-border);
          border-radius: 8px;
          padding: 12px 16px;
          font-family: 'JetBrains Mono', monospace;
          font-size: 18px;
          letter-spacing: 0.02em;
          display: flex;
          align-items: center;
          word-break: break-all;
        }
        .sb-cursor {
          display: inline-block;
          width: 9px; height: 20px;
          background: var(--accent);
          margin-left: 3px;
          animation: sb-blink 1s steps(1) infinite;
        }
        @media (prefers-reduced-motion: reduce) {
          .sb-cursor { animation: none; opacity: 0.6; }
        }
        @keyframes sb-blink { 50% { opacity: 0; } }

        .sb-actions {
          width: 100%;
          max-width: 720px;
          display: flex;
          justify-content: flex-end;
          margin-top: 14px;
        }
        .sb-clear-btn {
          font-family: 'Manrope', sans-serif;
          font-size: 13px;
          font-weight: 700;
          color: var(--text-secondary);
          background: transparent;
          border: 1px solid var(--panel-border);
          padding: 8px 16px;
          border-radius: 6px;
          cursor: pointer;
          transition: color 0.15s ease, border-color 0.15s ease;
        }
        .sb-clear-btn:hover { color: var(--text-primary); border-color: var(--accent); }
        .sb-clear-btn:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
      `}</style>

      <div className="sb-header">
        <h1 className="sb-title">Sign Buddy</h1>
        <div className="sb-status">
          <span className={`sb-status-dot ${status === "Connected" ? "is-live" : ""}`} />
          <span>{status}</span>
          {mode ? <span className="sb-mode">{mode}</span> : null}
        </div>
      </div>

      <div className="sb-video-frame">
        <video ref={videoRef} autoPlay playsInline muted style={{ display: "none" }} />
        <canvas ref={canvasRef} />
        <span className="sb-corner sb-corner--tl" />
        <span className="sb-corner sb-corner--tr" />
        <span className="sb-corner sb-corner--bl" />
        <span className="sb-corner sb-corner--br" />
      </div>

      <div className="sb-readout">
        <div className="sb-letter-tile">{letter || "—"}</div>
        <div className="sb-meter">
          <div className="sb-meter-label">
            <span>CONFIDENCE</span>
            <span className="sb-meter-value">
              {confidence != null ? `${Math.round(confidence * 100)}%` : "—"}
            </span>
          </div>
          <div className="sb-meter-track">
            <div
              className="sb-meter-fill"
              style={{ width: confidence != null ? `${Math.round(confidence * 100)}%` : "0%" }}
            />
          </div>
        </div>
      </div>

      <div className="sb-sentence-panel">
        <div className="sb-sentence-label">Spelled so far</div>
        <div className="sb-sentence-output">
          {sentence}
          <span className="sb-cursor" />
        </div>
      </div>

      <div className="sb-actions">
        <button className="sb-clear-btn" onClick={handleClear}>Clear</button>
      </div>
    </div>
  );
}
