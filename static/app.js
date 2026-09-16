/* vivo UI: mic -> WS (16 kHz int16 PCM) ; WS TTS (24 kHz int16 PCM) -> speaker.
 *
 * Protocol (see app/pipeline.py):
 *   client -> server : binary 16 kHz int16 mono PCM, {"type":"barge_in"|"flush"|"ping"}
 *   server -> client : JSON start|end|transcript|agent_text|tool|barge_ack|reply_done|error
 *                      + binary 24 kHz int16 mono TTS chunks (one per sentence)
 */
"use strict";

const TARGET_MIC_RATE = 16000;
const TARGET_PLAY_RATE = 24000;

/* Auto barge-in: while vivo is speaking, sustained mic level above the
 * threshold (your voice, after the browser's echo cancellation) interrupts
 * the reply. micLevel is rms*6 clamped to [0,1], sampled per ~25 ms mic
 * callback; typical speech lands 0.3-1.0, post-AEC echo usually < 0.2.
 * These are defaults only — the server sends its own values in the `config`
 * handshake message (vivo.toml [barge_in]) and overrides them. */
const BARGE_DEFAULTS = {
  levelThreshold: 0.25,
  sustainMs: 250, // level must stay above threshold this long
  cooldownMs: 700, // suppress re-triggers right after a barge
};

const $ = (id) => document.getElementById(id);
const el = {
  canvas: $("dot"),
  status: $("status"),
  statusLabel: $("status-label"),
  meterFill: $("meter-fill"),
  transcript: $("transcript"),
  btnStart: $("btn-start"),
  btnBarage: $("btn-barage"),
  btnFlush: $("btn-flush"),
  btnClear: $("btn-clear"),
  hint: $("hint"),
};

const S = {
  ws: null,
  wsState: "connecting", // connecting | open | closed
  pipeline: "idle", // idle | listening | thinking | speaking
  mic: null, // { stream, ctx, source, proc, rate }
  micActive: false,
  play: null, // { ctx, gain, analyser, nextPlayTime, sources:Set }
  micLevelTarget: 0,
  micLevel: 0,
  playLevel: 0,
  currentVivoEntry: null,
  pingTimer: null,
  bargeCfg: { ...BARGE_DEFAULTS }, // overridden by the server's `config` message
  bargePendingSince: null, // timestamp mic level started sustaining above threshold
  bargeCooldownUntil: 0, // suppress auto-barge re-triggers until this time
  dropAudio: false, // ignore in-flight TTS frames after a barge until the next `end`
};

/* ---------------- websocket ---------------- */

function connectWS() {
  setWsState("connecting");
  const url = `${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/ws`;
  const ws = new WebSocket(url);
  ws.binaryType = "arraybuffer";
  let wasOpen = false;
  ws.onopen = () => {
    wasOpen = true;
    setWsState("open");
    startPinging();
  };
  ws.onmessage = (ev) => {
    if (typeof ev.data === "string") handleServerJson(JSON.parse(ev.data));
    else enqueueTTS(ev.data);
  };
  ws.onclose = () => {
    setWsState("closed");
    stopPinging();
    if (wasOpen) addEntry("error", "disconnected from server");
  };
  S.ws = ws;
}

function sendJson(obj) {
  if (S.ws && S.ws.readyState === WebSocket.OPEN) S.ws.send(JSON.stringify(obj));
}

function startPinging() {
  stopPinging();
  S.pingTimer = setInterval(() => sendJson({ type: "ping" }), 15000);
}

function stopPinging() {
  if (S.pingTimer) clearInterval(S.pingTimer);
  S.pingTimer = null;
}

function setWsState(s) {
  S.wsState = s;
  updateStatus();
}

function setPipeline(s) {
  S.pipeline = s;
  if (s === "listening" || s === "thinking") S.currentVivoEntry = null;
  updateStatus();
}

/* ---------------- mic capture ---------------- */

function micUnavailableMsg() {
  return (
    "Microphone needs a secure context (HTTPS or localhost) &mdash; this page " +
    "is plain HTTP on a non-localhost host. Options: open the HTTPS version of " +
    "this page (e.g. via a reverse proxy with a valid certificate, or a " +
    "self-signed cert you accept once), open " +
    "<code>http://localhost:8600</code> from this machine, or in Chrome/Edge " +
    "enable <code>chrome://flags/#unsafely-treat-insecure-origin-as-secure</code> " +
    "for this origin and relaunch."
  );
}

async function startMic() {
  if (!navigator.mediaDevices) throw new Error("unavailable in this browser context");
  const stream = await navigator.mediaDevices.getUserMedia({
    audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: true, channelCount: 1 },
  });
  const ctx = new AudioContext();
  const source = ctx.createMediaStreamSource(stream);
  const proc = ctx.createScriptProcessor(4096, 1, 1);
  source.connect(proc);
  const mute = ctx.createGain();
  mute.gain.value = 0;
  proc.connect(mute);
  mute.connect(ctx.destination);

  proc.onaudioprocess = (e) => {
    const input = e.inputBuffer.getChannelData(0);
    let sum = 0;
    for (let i = 0; i < input.length; i++) sum += input[i] * input[i];
    const rms = Math.sqrt(sum / input.length);
    S.micLevelTarget = Math.min(1, rms * 6);
    checkAutoBarge();

    if (S.ws && S.ws.readyState !== WebSocket.OPEN) return;
    const resampled = resampleTo16k(input, ctx.sampleRate);
    S.ws.send(f32To16(resampled).buffer);
  };

  await ctx.resume();
  S.mic = { stream, ctx, source, proc };
  S.micActive = true;
  el.btnStart.textContent = "stop";
  el.btnStart.classList.add("active");
  el.hint.innerHTML = "mic on &middot; talk to me";
  updateStatus();
}

function stopMic() {
  if (S.mic) {
    try { S.mic.proc.disconnect(); } catch (_) {}
    try { S.mic.source.disconnect(); } catch (_) {}
    for (const t of S.mic.stream.getTracks()) t.stop();
    S.mic.ctx.close().catch(() => {});
  }
  S.mic = null;
  S.micActive = false;
  S.micLevelTarget = 0;
  el.btnStart.textContent = "start";
  el.btnStart.classList.remove("active");
  el.hint.innerHTML = "mic off &middot; click <b>start</b> to allow the microphone";
  updateStatus();
}

function resampleTo16k(input, fromRate) {
  if (fromRate === TARGET_MIC_RATE) return input;
  const ratio = fromRate / TARGET_MIC_RATE;
  const length = Math.floor(input.length / ratio);
  const out = new Float32Array(length);
  for (let i = 0; i < length; i++) {
    const pos = i * ratio;
    const idx = Math.floor(pos);
    const frac = pos - idx;
    const a = input[idx] || 0;
    const b = input[idx + 1] !== undefined ? input[idx + 1] : a;
    out[i] = a + (b - a) * frac;
  }
  return out;
}

function f32To16(f) {
  const out = new Int16Array(f.length);
  for (let i = 0; i < f.length; i++) {
    const s = Math.max(-1, Math.min(1, f[i]));
    out[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
  }
  return out;
}

/* ---------------- playback ---------------- */

function ensurePlay() {
  if (S.play) return;
  const ctx = new AudioContext();
  const gain = ctx.createGain();
  const analyser = ctx.createAnalyser();
  analyser.fftSize = 1024;
  gain.connect(analyser);
  analyser.connect(ctx.destination);
  S.play = { ctx, gain, analyser, nextPlayTime: 0, sources: new Set() };
}

function enqueueTTS(buf) {
  if (S.dropAudio) return; // stale frame from a barged-in reply
  ensurePlay();
  const p = S.play;
  const int16 = new Int16Array(buf);
  if (!int16.length) return;
  const float = new Float32Array(int16.length);
  for (let i = 0; i < int16.length; i++) float[i] = int16[i] / 32768;
  const audioBuf = p.ctx.createBuffer(1, float.length, TARGET_PLAY_RATE);
  audioBuf.copyToChannel(float, 0);

  const src = p.ctx.createBufferSource();
  src.buffer = audioBuf;
  src.connect(p.gain);
  const startAt = Math.max(p.ctx.currentTime, p.nextPlayTime);
  src.start(startAt);
  p.nextPlayTime = startAt + audioBuf.duration;
  p.sources.add(src);
  src.onended = () => p.sources.delete(src);
  setPipeline("speaking");
}

function stopPlayback() {
  if (!S.play) return;
  for (const src of S.play.sources) {
    try { src.stop(); } catch (_) {}
  }
  S.play.sources.clear();
  S.play.nextPlayTime = S.play.ctx.currentTime;
}

/* ---------------- barge-in ---------------- */

function vivoIsSpeaking() {
  return S.pipeline === "speaking" || (S.play && S.play.sources.size > 0);
}

function bargeIn() {
  if (!S.ws || S.ws.readyState !== WebSocket.OPEN) return;
  S.dropAudio = true; // TTS frames already in flight must not replay
  S.bargePendingSince = null;
  S.bargeCooldownUntil = performance.now() + S.bargeCfg.cooldownMs;
  stopPlayback();
  sendJson({ type: "barge_in" });
}

function checkAutoBarge() {
  const now = performance.now();
  if (!vivoIsSpeaking() || now < S.bargeCooldownUntil) {
    S.bargePendingSince = null;
    return;
  }
  if (S.micLevelTarget > S.bargeCfg.levelThreshold) {
    if (S.bargePendingSince === null) {
      S.bargePendingSince = now;
    } else if (now - S.bargePendingSince >= S.bargeCfg.sustainMs) {
      bargeIn();
    }
  } else {
    S.bargePendingSince = null; // dropped below threshold: restart the sustain window
  }
}

function readPlayLevel() {
  if (!S.play || S.play.sources.size === 0) return 0;
  const p = S.play;
  const arr = readPlayLevel.buf || (readPlayLevel.buf = new Float32Array(p.analyser.fftSize));
  p.analyser.getFloatTimeDomainData(arr);
  let sum = 0;
  for (let i = 0; i < arr.length; i++) sum += arr[i] * arr[i];
  return Math.min(1, Math.sqrt(sum / arr.length) * 5);
}

/* ---------------- server messages ---------------- */

function handleServerJson(m) {
  switch (m.type) {
    case "config":
      // server-side barge-in timings (vivo.toml [barge_in])
      if (m.barge_in) {
        S.bargeCfg = {
          levelThreshold: m.barge_in.level_threshold ?? S.bargeCfg.levelThreshold,
          sustainMs: m.barge_in.sustain_ms ?? S.bargeCfg.sustainMs,
          cooldownMs: m.barge_in.cooldown_ms ?? S.bargeCfg.cooldownMs,
        };
      }
      return; // config never touches the status UI
    case "start":
      setPipeline("listening");
      break;
    case "end":
      S.dropAudio = false; // a fresh reply: its audio is welcome again
      setPipeline("thinking");
      break;
    case "transcript":
      addEntry("you", m.text);
      setPipeline("thinking");
      break;
    case "agent_text":
      appendAgent(m.delta);
      break;
    case "tool":
      addEntry("tool", `${m.name} &rarr; ${escapeHtml(String(m.result).slice(0, 120))}`);
      break;
    case "barge_ack":
      S.dropAudio = true;
      stopPlayback(); // belt-and-braces: barge may not have come from our button
      setPipeline("idle");
      break;
    case "reply_done":
      setPipeline("idle");
      break;
    case "error":
      addEntry("error", m.message);
      setPipeline("idle");
      break;
    case "pong":
      break;
  }
  updateStatus();
}

function addEntry(cls, html) {
  const d = document.createElement("div");
  d.className = `entry entry-${cls}`;
  d.innerHTML = `<span class="who">${cls}</span><p></p>`;
  d.querySelector("p").innerHTML = html;
  el.transcript.appendChild(d);
  scrollTranscript();
  return d;
}

function appendAgent(delta) {
  let entry = S.currentVivoEntry;
  if (!entry) {
    entry = document.createElement("div");
    entry.className = "entry entry-vivo";
    entry.innerHTML = `<span class="who">vivo</span><p></p>`;
    el.transcript.appendChild(entry);
    S.currentVivoEntry = entry;
  }
  entry.querySelector("p").textContent += delta;
  scrollTranscript();
}

function scrollTranscript() {
  el.transcript.scrollTop = el.transcript.scrollHeight;
}

function escapeHtml(s) {
  return s.replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

/* ---------------- status ---------------- */

function updateStatus() {
  let state, label;
  if (S.wsState === "closed") [state, label] = ["closed", "Disconnected"];
  else if (S.wsState === "connecting") [state, label] = ["connecting", "Connecting…"];
  else if (S.play && S.play.sources.size > 0) [state, label] = ["speaking", "Speaking…"];
  else if (S.pipeline === "speaking") [state, label] = ["speaking", "Speaking…"];
  else if (S.pipeline === "listening") [state, label] = ["listening", "Listening…"];
  else if (S.pipeline === "thinking") [state, label] = ["thinking", "Thinking…"];
  else [state, label] = S.micActive ? ["ready", "Ready"] : ["ready", "Ready &mdash; mic off"];

  el.status.dataset.state = state;
  el.statusLabel.innerHTML = label;
}

/* ---------------- the dot ---------------- */

const canvas = el.canvas;
const c2d = canvas.getContext("2d");
let t = 0;

function resizeCanvas() {
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  canvas.width = Math.max(1, Math.floor(rect.width * dpr));
  canvas.height = Math.max(1, Math.floor(rect.height * dpr));
}

const STATE_COLORS = {
  connecting: ["#94a3b8", "#475569"],
  closed: ["#f87171", "#7f1d1d"],
  idle: ["#67e8f9", "#155e75"],
  listening: ["#86efac", "#15803d"],
  thinking: ["#93c5fd", "#1d4ed8"],
  speaking: ["#fde68a", "#b45309"],
};

function currentDotState() {
  if (S.wsState !== "open") return S.wsState;
  if (S.play && S.play.sources.size > 0) return "speaking";
  return S.pipeline === "idle" ? "idle" : S.pipeline;
}

function smoothLevel(prev, target) {
  const coef = target > prev ? 0.5 : 0.12;
  return prev + (target - prev) * coef;
}

function blinkScale(time) {
  const cycle = time % 3.6;
  if (cycle > 3.25) {
    const p = (cycle - 3.25) / 0.35;
    return 1 - 0.92 * Math.sin(Math.PI * p);
  }
  return 1;
}

function frame() {
  t += 1 / 60;
  S.micLevel = smoothLevel(S.micLevel, S.micLevelTarget);
  S.playLevel = smoothLevel(S.playLevel, readPlayLevel());

  const w = canvas.width, h = canvas.height;
  c2d.clearRect(0, 0, w, h);
  const cx = w / 2, cy = h / 2;

  const level = Math.max(S.micLevel, S.playLevel);
  const base = Math.min(w, h) * 0.17;
  const grow = 1 + level * 0.85;
  const R = base * grow;
  const wobbleAmt = 0.04 + level * 0.30;
  const [inner, outer] = STATE_COLORS[currentDotState()];

  const N = 96;
  c2d.beginPath();
  for (let i = 0; i <= N; i++) {
    const th = (i / N) * Math.PI * 2;
    const wob =
      0.6 * Math.sin(3 * th + t * 1.3) +
      0.4 * Math.sin(5 * th - t * 1.9 + 1.0) +
      0.3 * Math.sin(2 * th + t * 0.7);
    const r = R * (1 + wobbleAmt * wob);
    const x = cx + r * Math.cos(th);
    const y = cy + r * Math.sin(th);
    if (i === 0) c2d.moveTo(x, y);
    else c2d.lineTo(x, y);
  }
  c2d.closePath();
  const grad = c2d.createRadialGradient(cx - R * 0.35, cy - R * 0.4, R * 0.1, cx, cy, R * 1.5);
  grad.addColorStop(0, inner);
  grad.addColorStop(1, outer);
  c2d.fillStyle = grad;
  c2d.shadowColor = outer;
  c2d.shadowBlur = 40 * (0.4 + level);
  c2d.fill();
  c2d.shadowBlur = 0;

  // eyes
  const blink = blinkScale(t);
  const eyeR = R * 0.12;
  const eyeDX = R * 0.34, eyeDY = -R * 0.08;
  for (const side of [-1, 1]) {
    c2d.save();
    c2d.translate(cx + side * eyeDX, cy + eyeDY);
    c2d.scale(1, Math.max(0.08, blink));
    c2d.beginPath();
    c2d.fillStyle = "rgba(10, 10, 16, 0.9)";
    c2d.arc(0, 0, eyeR, 0, Math.PI * 2);
    c2d.fill();
    c2d.beginPath();
    c2d.fillStyle = "rgba(255, 255, 255, 0.85)";
    c2d.arc(-eyeR * 0.3, -eyeR * 0.3, eyeR * 0.25, 0, Math.PI * 2);
    c2d.fill();
    c2d.restore();
  }

  el.meterFill.style.width = `${Math.round(S.micLevel * 100)}%`;
  requestAnimationFrame(frame);
}

/* ---------------- wiring ---------------- */

el.btnStart.addEventListener("click", async () => {
  try {
    if (S.micActive) stopMic();
    else await startMic();
  } catch (err) {
    if (!navigator.mediaDevices) addEntry("error", micUnavailableMsg());
    else addEntry("error", `microphone: ${escapeHtml(err.message || String(err))}`);
  }
});

el.btnBarage.addEventListener("click", bargeIn);

el.btnFlush.addEventListener("click", () => sendJson({ type: "flush" }));

el.btnClear.addEventListener("click", () => {
  el.transcript.innerHTML = "";
  S.currentVivoEntry = null;
});

window.addEventListener("resize", resizeCanvas);
window.addEventListener("load", () => {
  resizeCanvas();
  connectWS();
  requestAnimationFrame(frame);
  if (!window.isSecureContext || !navigator.mediaDevices) {
    el.btnStart.disabled = true;
    el.hint.innerHTML = "mic blocked &mdash; not a secure context (see sidebar)";
    addEntry("error", micUnavailableMsg());
  }
});
