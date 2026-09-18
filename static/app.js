/* vivo UI: mic -> WS (16 kHz int16 PCM) ; WS TTS (48 kHz int16 PCM by default) -> speaker.
 *
 * Protocol (see app/pipeline.py):
 *   client -> server : binary 16 kHz int16 PCM,
 *                      {"type":"barge_in"|"flush"|"ping"|{"type":"session"[,"id":str]}}
 *   server -> client : JSON start|end|transcript|agent_text|tool|barge_ack|reply_done|error|session|config
 *                      + binary int16 mono TTS chunks (one per sentence; sample rate via config.audio.tts_sample_rate)
 */
"use strict";

const TARGET_MIC_RATE = 16000;
const DEFAULT_PLAY_RATE = 48000;

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
  btnDream: $("btn-dream"),
  btnClear: $("btn-clear"),
  sessionSelect: $("session-select"),
  btnNewSession: $("btn-new-session"),
  btnRenameSession: $("btn-rename-session"),
  btnDeleteSession: $("btn-delete-session"),
  btnSettings: $("btn-settings"),
  motionPreset: $("motion-preset"),
  settingsDlg: $("settings"),
  settingsBody: $("settings-body"),
  settingsMsg: $("settings-msg"),
  btnSettingsSave: $("btn-settings-save"),
  btnSettingsClose: $("btn-settings-close"),
  btnSettingsX: $("btn-settings-x"),
  hint: $("hint"),
};

const S = {
  ws: null,
  wsState: "connecting", // connecting | open | closed
  sessionId: null, // conversation bound to this tab (localStorage "vivo.session")
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
  ttsSampleRate: DEFAULT_PLAY_RATE,
  dreaming: false,
  motionPreset: "balanced",
  prefersReducedMotion: false,
  blink: { active: false, startedAt: 0, duration: 0, nextAt: 0, doubleBlink: false },
  hydrateToken: 0,
  sessionsById: {},
};

try { S.sessionId = localStorage.getItem("vivo.session"); } catch (_) { S.sessionId = null; }

function rememberSession(id) {
  S.sessionId = id;
  try { localStorage.setItem("vivo.session", id); } catch (_) {}
}

/* ---------------- websocket ---------------- */

function connectWS() {
  setWsState("connecting");
  const base = `${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/ws`;
  const url = S.sessionId ? `${base}?session=${encodeURIComponent(S.sessionId)}` : base;
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
  const rate = Number.isFinite(S.ttsSampleRate) ? S.ttsSampleRate : DEFAULT_PLAY_RATE;
  const audioBuf = p.ctx.createBuffer(1, float.length, rate);
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
      if (m.audio) {
        const sr = Number(m.audio.tts_sample_rate);
        if (Number.isFinite(sr) && sr >= 8000 && sr <= 96000) S.ttsSampleRate = sr;
      }
      return; // config never touches the status UI
    case "dream":
      S.dreaming = !!m.active;
      updateStatus();
      return;
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
    case "reminder":
      const reminderText = String(m.text || "Reminder");
      addEntry("hint", `⏰ reminder: ${escapeHtml(reminderText)}`);
      return;
    case "tool":
      addToolEntry(m.name, m.result);
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
    case "session":
      // handshake + result of new/switch commands — the server's word on which
      // conversation this tab is bound to
      rememberSession(m.id);
      el.sessionSelect.disabled = false;
      loadSessions().then(() => {
        el.sessionSelect.value = m.id;
      });
      hydrateTranscript(m.id);
      if (m.created) addEntry("hint", "&mdash; new conversation &mdash;");
      return; // never touches the status UI
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

function addToolEntry(name, result) {
  const d = document.createElement("details");
  d.className = "entry entry-tool entry-tool-details";

  const summary = document.createElement("summary");
  summary.className = "tool-summary";

  const who = document.createElement("span");
  who.className = "who";
  who.textContent = "tool";
  summary.appendChild(who);

  const toolName = document.createElement("span");
  toolName.className = "tool-name";
  toolName.textContent = name || "unknown";
  summary.appendChild(toolName);

  const toolHint = document.createElement("span");
  toolHint.className = "tool-hint";
  toolHint.textContent = "click to expand";
  summary.appendChild(toolHint);

  const body = document.createElement("div");
  body.className = "tool-body";
  const pre = document.createElement("pre");
  pre.textContent = String(result ?? "");
  body.appendChild(pre);

  d.appendChild(summary);
  d.appendChild(body);
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

async function hydrateTranscript(sessionId) {
  if (!sessionId) return;
  const token = ++S.hydrateToken;
  let j;
  try {
    const res = await fetch(`/api/sessions/${encodeURIComponent(sessionId)}/transcript`);
    if (!res.ok) return;
    j = await res.json();
  } catch (_) {
    return;
  }
  if (token !== S.hydrateToken || sessionId !== S.sessionId) return;

  el.transcript.innerHTML = "";
  S.currentVivoEntry = null;

  if (j.summary) {
    addEntry("hint", `Earlier summary: ${escapeHtml(String(j.summary))}`);
  }
  for (const turn of j.turns || []) {
    if (turn.user) addEntry("you", escapeHtml(String(turn.user)));
    if (turn.assistant) addEntry("vivo", escapeHtml(String(turn.assistant)));
  }
}

/* ---------------- sessions (T021) ----------------
 * Each tab binds to one named conversation (by default the server's active
 * one) and remembers it in localStorage. "new" starts a fresh conversation,
 * the dropdown switches; both go over WS so every tab agrees (the server
 * answers each command with a `session` frame).
 */

const MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];

function formatSessionId(id) {
  const m = /^(\d{4})(\d{2})(\d{2})-(\d{2})(\d{2})(-.*)?$/.exec(id);
  if (!m) return id;
  const label = `${MONTHS[Number(m[2]) - 1]} ${m[3]} ${m[4]}:${m[5]}`;
  return m[6] ? `${label}${m[6]}` : label;
}

async function loadSessions() {
  let j;
  try {
    const res = await fetch("/api/sessions");
    if (!res.ok) return;
    j = await res.json();
  } catch (_) {
    return; // server unreachable — the WS handshake will still bind us
  }
  const sel = el.sessionSelect;
  sel.innerHTML = "";
  S.sessionsById = {};
  for (const s of j.sessions) {
    S.sessionsById[s.id] = s;
    const opt = document.createElement("option");
    opt.value = s.id;
    const title = (s.name || "").trim() || formatSessionId(s.id);
    opt.textContent = title + (s.active ? " (active)" : "");
    sel.appendChild(opt);
  }
  const known = S.sessionId && [...sel.options].some((o) => o.value === S.sessionId);
  if (!known) S.sessionId = j.active; // remembered session vanished: follow the server
  if (S.sessionId) sel.value = S.sessionId;
  sel.disabled = false;
}

async function renameSelectedSession() {
  const id = el.sessionSelect.value;
  if (!id) return;
  const meta = S.sessionsById[id] || {};
  const current = (meta.name || "").trim();
  const fallback = formatSessionId(id);
  const name = prompt(
    `Rename conversation (${fallback}). Leave blank to reset to auto label.`,
    current
  );
  if (name === null) return;

  let res;
  try {
    res = await fetch(`/api/sessions/${encodeURIComponent(id)}/rename`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name }),
    });
  } catch (err) {
    addEntry("error", `rename failed: ${escapeHtml(err.message || String(err))}`);
    return;
  }
  if (!res.ok) {
    let msg = `HTTP ${res.status}`;
    try {
      const j = await res.json();
      if (j && j.detail) msg = String(j.detail);
    } catch (_) {}
    addEntry("error", `rename failed: ${escapeHtml(msg)}`);
    return;
  }
  await loadSessions();
  if (S.sessionId) el.sessionSelect.value = S.sessionId;
}

async function deleteSelectedSession() {
  const id = el.sessionSelect.value;
  if (!id) return;
  const meta = S.sessionsById[id] || {};
  const title = (meta.name || "").trim() || formatSessionId(id);
  if (!confirm(`Delete conversation "${title}"? This cannot be undone.`)) return;

  let res;
  try {
    res = await fetch(`/api/sessions/${encodeURIComponent(id)}`, { method: "DELETE" });
  } catch (err) {
    addEntry("error", `delete failed: ${escapeHtml(err.message || String(err))}`);
    return;
  }
  if (!res.ok) {
    let msg = `HTTP ${res.status}`;
    try {
      const j = await res.json();
      if (j && j.detail) msg = String(j.detail);
    } catch (_) {}
    addEntry("error", `delete failed: ${escapeHtml(msg)}`);
    return;
  }

  let active = null;
  try {
    const j = await res.json();
    active = j && j.active;
  } catch (_) {}
  await loadSessions();
  if (active) sendJson({ type: "session", id: active });
}

/* ---------------- status ---------------- */

function updateStatus() {
  let state, label;
  if (S.wsState === "closed") [state, label] = ["closed", "Disconnected"];
  else if (S.wsState === "connecting") [state, label] = ["connecting", "Connecting…"];
  else if (S.dreaming) [state, label] = ["dreaming", "Dreaming…"];
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
  dreaming: ["#c4b5fd", "#6d28d9"],
  speaking: ["#fde68a", "#b45309"],
};

const MOTION_PRESETS = {
  calm: {
    levelGrow: 0.35,
    wobbleBase: 0.02,
    wobbleLevel: 0.11,
    bubbleDrift: 0.014,
    mouthOpenBase: 0.045,
    mouthOpenLevel: 0.12,
    mouthJitter: 0.006,
  },
  balanced: {
    levelGrow: 0.55,
    wobbleBase: 0.03,
    wobbleLevel: 0.16,
    bubbleDrift: 0.022,
    mouthOpenBase: 0.06,
    mouthOpenLevel: 0.17,
    mouthJitter: 0.009,
  },
  expressive: {
    levelGrow: 0.75,
    wobbleBase: 0.045,
    wobbleLevel: 0.24,
    bubbleDrift: 0.033,
    mouthOpenBase: 0.08,
    mouthOpenLevel: 0.23,
    mouthJitter: 0.013,
  },
};

const MOTION = { ...MOTION_PRESETS.balanced };
const VISUAL_TRANSITION_MS = 260;
const BLINK = {
  gapMinMs: 2600,
  gapMaxMs: 7800,
  durationMinMs: 150,
  durationMaxMs: 240,
  closedHoldMinMs: 30,
  closedHoldMaxMs: 70,
  doubleChance: 0.18,
  doubleGapMinMs: 140,
  doubleGapMaxMs: 260,
};
const visual = {
  from: "idle",
  to: "idle",
  startedAt: performance.now(),
};

function clamp01(v) {
  return Math.max(0, Math.min(1, v));
}

function randRange(min, max) {
  return min + Math.random() * (max - min);
}

function lerp(a, b, t) {
  return a + (b - a) * t;
}

function hexToRgb(hex) {
  const h = hex.replace("#", "");
  return {
    r: parseInt(h.slice(0, 2), 16),
    g: parseInt(h.slice(2, 4), 16),
    b: parseInt(h.slice(4, 6), 16),
  };
}

function mixHex(a, b, t) {
  const aa = hexToRgb(a);
  const bb = hexToRgb(b);
  const r = Math.round(lerp(aa.r, bb.r, t));
  const g = Math.round(lerp(aa.g, bb.g, t));
  const bl = Math.round(lerp(aa.b, bb.b, t));
  return `rgb(${r}, ${g}, ${bl})`;
}

function visualProgress(now) {
  return clamp01((now - visual.startedAt) / VISUAL_TRANSITION_MS);
}

function updateVisualTarget(nextState, now) {
  if (visual.to === nextState) return;
  visual.from = visual.to;
  visual.to = nextState;
  visual.startedAt = now;
}

function stateWeight(name, p) {
  if (visual.from === visual.to) return visual.to === name ? 1 : 0;
  const fromW = visual.from === name ? 1 - p : 0;
  const toW = visual.to === name ? p : 0;
  return fromW + toW;
}

function applyMotionPreset(name, persist = true) {
  const key = Object.prototype.hasOwnProperty.call(MOTION_PRESETS, name) ? name : "balanced";
  S.motionPreset = key;
  if (el.motionPreset) el.motionPreset.value = key;

  const scale = S.prefersReducedMotion ? 0.62 : 1;
  const src = MOTION_PRESETS[key];
  for (const k of Object.keys(MOTION)) MOTION[k] = src[k] * scale;

  if (persist) {
    try { localStorage.setItem("vivo.motionPreset", key); } catch (_) {}
  }
}

function setReducedMotion(enabled) {
  S.prefersReducedMotion = !!enabled;
  applyMotionPreset(S.motionPreset, false);
}

function initMotionControls() {
  let saved = null;
  try { saved = localStorage.getItem("vivo.motionPreset"); } catch (_) {}
  S.motionPreset = Object.prototype.hasOwnProperty.call(MOTION_PRESETS, saved) ? saved : "balanced";

  if (window.matchMedia) {
    const mq = window.matchMedia("(prefers-reduced-motion: reduce)");
    setReducedMotion(mq.matches);
    const onChange = (ev) => setReducedMotion(ev.matches);
    if (mq.addEventListener) mq.addEventListener("change", onChange);
    else if (mq.addListener) mq.addListener(onChange);
  } else {
    setReducedMotion(false);
  }

  applyMotionPreset(S.motionPreset, false);
}

function currentDotState() {
  if (S.wsState !== "open") return S.wsState;
  if (S.dreaming) return "dreaming";
  if (S.play && S.play.sources.size > 0) return "speaking";
  return S.pipeline === "idle" ? "idle" : S.pipeline;
}

function smoothLevel(prev, target) {
  const coef = target > prev ? 0.5 : 0.12;
  return prev + (target - prev) * coef;
}

function resetBlinkSchedule(now = performance.now()) {
  S.blink.active = false;
  S.blink.startedAt = 0;
  S.blink.duration = 0;
  S.blink.closedHold = 0;
  S.blink.doubleBlink = false;
  S.blink.nextAt = now + randRange(BLINK.gapMinMs, BLINK.gapMaxMs);
}

function blinkScale(now) {
  const blink = S.blink;
  if (!blink.nextAt) resetBlinkSchedule(now);

  if (!blink.active && now >= blink.nextAt) {
    blink.active = true;
    blink.startedAt = now;
    blink.duration = randRange(BLINK.durationMinMs, BLINK.durationMaxMs);
    blink.closedHold = randRange(BLINK.closedHoldMinMs, BLINK.closedHoldMaxMs);
    blink.doubleBlink = Math.random() < BLINK.doubleChance;
  }

  if (!blink.active) return 1;

  const elapsed = now - blink.startedAt;
  const closeDur = blink.duration * 0.42;
  const openDur = blink.duration * 0.58;
  let scale = 1;

  if (elapsed < closeDur) {
    const p = clamp01(elapsed / closeDur);
    const eased = 1 - Math.cos((Math.PI * p) / 2);
    scale = 1 - 0.92 * eased;
  } else if (elapsed < closeDur + blink.closedHold) {
    scale = 0.08;
  } else if (elapsed < blink.duration) {
    const p = clamp01((elapsed - closeDur - blink.closedHold) / openDur);
    const eased = Math.sin((Math.PI * p) / 2);
    scale = 0.08 + 0.92 * eased;
  }

  if (elapsed >= blink.duration) {
    if (blink.doubleBlink) {
      blink.active = false;
      blink.startedAt = 0;
      blink.duration = 0;
      blink.closedHold = 0;
      blink.doubleBlink = false;
      blink.nextAt = now + randRange(BLINK.doubleGapMinMs, BLINK.doubleGapMaxMs);
    } else {
      resetBlinkSchedule(now);
    }
  }

  return scale;
}

function drawIdleBreath(cx, cy, R, time, weight) {
  if (weight < 0.02) return;
  const pulse = 0.5 + 0.5 * Math.sin(time * 1.05);
  const scale = 1 + 0.018 * pulse * weight;
  const glow = 0.06 + 0.08 * pulse * weight;

  c2d.save();
  c2d.globalAlpha = glow;
  c2d.strokeStyle = "rgba(255, 255, 255, 0.9)";
  c2d.lineWidth = Math.max(1.5, R * 0.02);
  c2d.beginPath();
  c2d.arc(cx, cy, R * scale * 0.82, 0, Math.PI * 2);
  c2d.stroke();
  c2d.globalAlpha = glow * 0.7;
  c2d.beginPath();
  c2d.arc(cx, cy, R * scale * 0.64, 0, Math.PI * 2);
  c2d.stroke();
  c2d.restore();
}

function drawListeningRings(cx, cy, R, time, weight) {
  if (weight < 0.02) return;
  const pulse = 0.5 + 0.5 * Math.sin(time * 2.2);
  const spin = time * 1.25;
  const rings = [0.84, 1.04];

  c2d.save();
  c2d.lineCap = "round";
  c2d.lineWidth = Math.max(1.6, R * 0.018);
  for (let i = 0; i < rings.length; i++) {
    const rr = R * rings[i];
    const start = spin + i * 0.95;
    const span = Math.PI * (0.5 + 0.14 * pulse);
    c2d.strokeStyle = `rgba(102, 255, 190, ${0.15 + 0.12 * weight})`;
    c2d.beginPath();
    c2d.arc(cx, cy, rr, start, start + span);
    c2d.stroke();
  }

  const dots = [
    { a: spin * 0.9, r: R * 1.08 },
    { a: spin * 0.9 + 2.1, r: R * 1.08 },
  ];
  for (const dot of dots) {
    const x = cx + Math.cos(dot.a) * dot.r;
    const y = cy + Math.sin(dot.a) * dot.r;
    c2d.beginPath();
    c2d.fillStyle = `rgba(176, 255, 225, ${(0.12 + 0.12 * pulse) * weight})`;
    c2d.arc(x, y, Math.max(1.4, R * 0.028), 0, Math.PI * 2);
    c2d.fill();
  }
  c2d.restore();
}

function drawConnectingOrbit(cx, cy, R, time, weight) {
  if (weight < 0.02) return;
  const spin = time * 1.8;
  const orbit = R * 1.08;
  const dash = R * 0.22;

  c2d.save();
  c2d.lineCap = "round";
  c2d.setLineDash([dash, dash * 0.7]);
  c2d.lineDashOffset = -spin * 12;
  c2d.strokeStyle = `rgba(200, 220, 255, ${0.12 + 0.15 * weight})`;
  c2d.lineWidth = Math.max(1.5, R * 0.018);
  c2d.beginPath();
  c2d.arc(cx, cy, orbit, 0, Math.PI * 2);
  c2d.stroke();
  c2d.setLineDash([]);

  for (let i = 0; i < 3; i++) {
    const a = spin + i * (Math.PI * 2 / 3);
    const x = cx + Math.cos(a) * orbit;
    const y = cy + Math.sin(a) * orbit;
    c2d.beginPath();
    c2d.fillStyle = `rgba(255, 255, 255, ${(0.2 + 0.2 * Math.sin(time * 2.8 + i)) * weight})`;
    c2d.arc(x, y, Math.max(1.6, R * 0.024), 0, Math.PI * 2);
    c2d.fill();
  }
  c2d.restore();
}

function drawClosedFlicker(cx, cy, R, time, weight) {
  if (weight < 0.02) return;
  const flicker = 0.5 + 0.5 * Math.sin(time * 3.7);
  const alpha = (0.08 + 0.08 * flicker) * weight;

  c2d.save();
  c2d.globalAlpha = alpha;
  c2d.strokeStyle = "rgba(255, 120, 126, 0.95)";
  c2d.lineWidth = Math.max(1.5, R * 0.02);
  c2d.beginPath();
  c2d.moveTo(cx - R * 0.35, cy - R * 0.32);
  c2d.lineTo(cx + R * 0.35, cy + R * 0.32);
  c2d.moveTo(cx + R * 0.35, cy - R * 0.32);
  c2d.lineTo(cx - R * 0.35, cy + R * 0.32);
  c2d.stroke();
  c2d.restore();
}

function drawThinkingBubbles(cx, cy, R, time, weight) {
  if (weight < 0.02) return;
  const phase = time * 0.95;
  const baseX = cx + R * 0.58;
  const baseY = cy - R * 0.72;
  const bubbles = [
    { d: 0.0, x: 0.00, y: 0.00, r: 0.12, a: 0.34 },
    { d: 0.7, x: 0.22, y: -0.24, r: 0.17, a: 0.30 },
    { d: 1.4, x: 0.44, y: -0.48, r: 0.22, a: 0.26 },
  ];

  c2d.save();
  for (const b of bubbles) {
    const k = (Math.sin(phase + b.d) + 1) * 0.5;
    const alpha = b.a * (0.65 + 0.35 * k) * weight;
    const bob = Math.sin(phase * 0.9 + b.d) * R * MOTION.bubbleDrift;
    const x = baseX + R * b.x;
    const y = baseY + R * b.y + bob;
    const r = R * b.r * (0.9 + 0.2 * k);

    c2d.beginPath();
    c2d.fillStyle = `rgba(220, 236, 255, ${alpha.toFixed(3)})`;
    c2d.arc(x, y, r, 0, Math.PI * 2);
    c2d.fill();

    c2d.beginPath();
    c2d.fillStyle = `rgba(255, 255, 255, ${(alpha * 0.65).toFixed(3)})`;
    c2d.arc(x - r * 0.28, y - r * 0.28, r * 0.24, 0, Math.PI * 2);
    c2d.fill();
  }
  c2d.restore();
}

function drawDreamingZs(cx, cy, R, time, weight) {
  if (weight < 0.02) return;
  const baseX = cx - R * 0.05;
  const baseY = cy - R * 1.18;
  const drift = Math.sin(time * 2.2) * 8;
  const zzzs = ["z", "zz", "zzz"];

  c2d.save();
  c2d.font = `${Math.max(14, R * 0.18)}px "Outfit", sans-serif`;
  c2d.textAlign = "center";
  c2d.textBaseline = "middle";
  for (let i = 0; i < zzzs.length; i++) {
    const ky = baseY - i * (R * 0.12) + drift * (i * 0.2 + 0.2);
    const kx = baseX + i * (R * 0.1);
    c2d.fillStyle = `rgba(196, 181, 253, ${(0.45 + 0.35 * Math.sin(time * 2.5 + i)) * weight})`;
    c2d.fillText(zzzs[i], kx, ky);
  }
  c2d.restore();
}

function drawMouth(cx, cy, R, speakingWeight, thinkingWeight, playLevel, time) {
  const speaking = speakingWeight > 0.03;
  const mouthY = cy + R * 0.33;
  const halfW = R * 0.22;
  const jitter = speaking ? Math.sin(time * 15) * R * MOTION.mouthJitter * speakingWeight : 0;
  const open = R * 0.022 + (R * (MOTION.mouthOpenBase + playLevel * MOTION.mouthOpenLevel) * speakingWeight) + Math.abs(jitter);
  const smile = R * (0.02 - 0.06 * thinkingWeight);

  c2d.save();
  c2d.lineCap = "round";

  // outer lip
  c2d.beginPath();
  c2d.strokeStyle = "rgba(10, 10, 16, 0.85)";
  c2d.lineWidth = Math.max(2, R * 0.045);
  c2d.moveTo(cx - halfW, mouthY);
  c2d.quadraticCurveTo(cx, mouthY + open + smile, cx + halfW, mouthY);
  c2d.stroke();

  // mouth cavity when speaking
  if (speaking) {
    c2d.beginPath();
    c2d.fillStyle = `rgba(12, 8, 12, ${0.25 + 0.42 * speakingWeight})`;
    c2d.moveTo(cx - halfW * 0.78, mouthY + R * 0.01);
    c2d.quadraticCurveTo(cx, mouthY + open * 1.18, cx + halfW * 0.78, mouthY + R * 0.01);
    c2d.quadraticCurveTo(cx, mouthY + R * 0.028, cx - halfW * 0.78, mouthY + R * 0.01);
    c2d.fill();
  }

  c2d.restore();
}

function drawSpeakingWave(cx, cy, R, time, weight) {
  if (weight < 0.02) return;
  const pulse = 0.5 + 0.5 * Math.sin(time * 2.7);
  const x = cx + R * (0.36 + 0.05 * pulse);
  const y = cy + R * 0.03;

  c2d.save();
  c2d.strokeStyle = `rgba(255, 244, 214, ${(0.08 + 0.15 * pulse) * weight})`;
  c2d.lineWidth = Math.max(1.5, R * 0.018);
  c2d.lineCap = "round";
  c2d.beginPath();
  c2d.arc(x, y, R * (0.16 + 0.03 * pulse), -0.8, 0.8);
  c2d.stroke();
  c2d.beginPath();
  c2d.arc(x + R * 0.08, y, R * (0.24 + 0.03 * pulse), -0.82, 0.82);
  c2d.stroke();
  c2d.restore();
}

function frame() {
  t += 1 / 60;
  S.micLevel = smoothLevel(S.micLevel, S.micLevelTarget);
  S.playLevel = smoothLevel(S.playLevel, readPlayLevel());

  const now = performance.now();

  const w = canvas.width, h = canvas.height;
  c2d.clearRect(0, 0, w, h);
  const cx = w / 2, cy = h / 2;

  const level = Math.max(S.micLevel, S.playLevel);
  const base = Math.min(w, h) * 0.17;
  const grow = 1 + level * MOTION.levelGrow;
  const R = base * grow;
  const wobbleAmt = MOTION.wobbleBase + level * MOTION.wobbleLevel;
  const dotState = currentDotState();
  updateVisualTarget(dotState, now);
  const progress = visualProgress(now);
  const fromColors = STATE_COLORS[visual.from] || STATE_COLORS.idle;
  const toColors = STATE_COLORS[visual.to] || STATE_COLORS.idle;
  const inner = mixHex(fromColors[0], toColors[0], progress);
  const outer = mixHex(fromColors[1], toColors[1], progress);
  const speakingWeight = stateWeight("speaking", progress);
  const thinkingWeight = stateWeight("thinking", progress);
  const dreamingWeight = stateWeight("dreaming", progress);

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

  const idleWeight = stateWeight("idle", progress);
  const listeningWeight = stateWeight("listening", progress);
  const connectingWeight = stateWeight("connecting", progress);
  const closedWeight = stateWeight("closed", progress);

  drawIdleBreath(cx, cy, R, t, idleWeight);
  drawListeningRings(cx, cy, R, t, listeningWeight);
  drawConnectingOrbit(cx, cy, R, t, connectingWeight);
  drawClosedFlicker(cx, cy, R, t, closedWeight);

  // eyes
  const blink = blinkScale(now);
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

  drawMouth(cx, cy, R, speakingWeight, thinkingWeight, S.playLevel, t);
  drawSpeakingWave(cx, cy, R, t, speakingWeight);
  drawThinkingBubbles(cx, cy, R, t, thinkingWeight);
  drawDreamingZs(cx, cy, R, t, dreamingWeight);

  el.meterFill.style.width = `${Math.round(S.micLevel * 100)}%`;
  requestAnimationFrame(frame);
}

/* ---------------- settings pane (T019) ----------------
 * Rendered entirely from the server's schema (GET /api/config): sliders for
 * numerics, dropdowns for enum/voice, checkboxes, textareas for phrases.
 * Saving POSTs the whole snapshot; the server validates, rewrites vivo.toml,
 * hot-applies what it can, and re-sends the `config` WS frame to open tabs.
 */

const APPLY_LABELS = { now: "live", next: "next session", restart: "restart" };
const APPLY_TIPS = {
  now: "Applies immediately, even mid-reply",
  next: "Applies from the next utterance or connection",
  restart: "Needs a container restart",
};

const settings = { values: null, schema: null, voices: [] };
let settingsMsgTimer = null;

function flashSettingsMsg(text, isError) {
  el.settingsMsg.textContent = text;
  el.settingsMsg.className = "settings-msg" + (isError ? " err" : " ok");
  if (settingsMsgTimer) clearTimeout(settingsMsgTimer);
  settingsMsgTimer = setTimeout(() => {
    el.settingsMsg.textContent = "";
    el.settingsMsg.className = "settings-msg";
  }, 3000);
}

async function triggerDream() {
  if (!S.ws || S.ws.readyState !== WebSocket.OPEN) {
    addEntry("error", "server disconnected — dream couldn't start");
    return;
  }
  try {
    const res = await fetch("/api/dream", { method: "POST" });
    if (!res.ok) {
      const j = await res.json().catch(() => ({}));
      throw new Error((j && j.detail) || `HTTP ${res.status}`);
    }
    const j = await res.json();
    if (j.summary && j.summary !== "No important memory yet.") {
      addEntry("hint", `dreamed: ${escapeHtml(String(j.summary))}`);
    } else {
      addEntry("hint", "dreamed: nothing notable yet");
    }
  } catch (err) {
    addEntry("error", `dream failed: ${escapeHtml(err.message || String(err))}`);
  }
}

async function openSettings() {
  el.settingsMsg.textContent = "";
  el.settingsMsg.className = "settings-msg";
  el.settingsBody.innerHTML = '<p class="settings-msg">Loading…</p>';
  el.settingsDlg.showModal();
  let j;
  try {
    const res = await fetch("/api/config");
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    j = await res.json();
  } catch (err) {
    el.settingsBody.innerHTML =
      `<p class="settings-msg err">could not load settings: ${escapeHtml(err.message)}</p>`;
    return;
  }
  settings.values = j.values;
  settings.schema = j.schema;
  settings.voices = j.voices;
  el.settingsBody.innerHTML = "";
  for (const [sec, spec] of Object.entries(j.schema)) {
    const section = document.createElement("section");
    section.className = "ssection";
    const h = document.createElement("h3");
    h.textContent = spec.title;
    section.appendChild(h);
    for (const [key, k] of Object.entries(spec.keys)) {
      section.appendChild(buildSettingRow(sec, key, k));
    }
    el.settingsBody.appendChild(section);
  }
}

function buildSettingRow(sec, key, k) {
  const value = settings.values[sec][key];
  const row = document.createElement("div");
  row.className = "srow";
  row.dataset.sec = sec;
  row.dataset.key = key;
  row.dataset.type = k.type;

  const label = document.createElement("div");
  label.className = "slabel";
  const name = document.createElement("span");
  name.textContent = k.label;
  label.appendChild(name);
  const tip = document.createElement("span");
  tip.className = "tip";
  tip.dataset.tip = k.help || "";
  tip.textContent = "?";
  label.appendChild(tip);
  const badge = document.createElement("span");
  badge.className = `sapply sapply-${k.apply}`;
  badge.textContent = APPLY_LABELS[k.apply] || k.apply;
  badge.title = APPLY_TIPS[k.apply] || k.apply;
  label.appendChild(badge);
  row.appendChild(label);

  const ctrlWrap = document.createElement("div");
  ctrlWrap.className = "sctrl";
  ctrlWrap.appendChild(buildControl(k, value));
  row.appendChild(ctrlWrap);
  return row;
}

function buildControl(k, value) {
  if (k.type === "int" || k.type === "float") {
    const range = document.createElement("input");
    range.type = "range";
    range.min = k.min;
    range.max = k.max;
    range.step = k.step;
    range.value = value;
    const val = document.createElement("span");
    val.className = "sval";
    val.textContent = value;
    range.addEventListener("input", () => { val.textContent = range.value; });
    const wrap = document.createElement("div");
    wrap.className = "srange";
    wrap.appendChild(range);
    wrap.appendChild(val);
    return wrap;
  }
  if (k.type === "bool") {
    const box = document.createElement("input");
    box.type = "checkbox";
    box.checked = !!value;
    return box;
  }
  if (k.type === "choices") {
    const sel = document.createElement("select");
    const opts = [...k.choices];
    if (!opts.includes(value)) opts.unshift(value);
    for (const o of opts) {
      const opt = document.createElement("option");
      opt.value = o;
      opt.textContent = o;
      if (o === value) opt.selected = true;
      sel.appendChild(opt);
    }
    return sel;
  }
  if (k.type === "voices") return buildVoiceControl(value);
  if (k.type === "textarea" || k.type === "str[]") {
    const ta = document.createElement("textarea");
    ta.rows = k.type === "str[]" ? 4 : 2;
    ta.value = k.type === "str[]" ? value.join("\n") : value;
    return ta;
  }
  const inp = document.createElement("input");
  inp.type = "text";
  inp.value = value;
  return inp;
}

/* Voice-clone controls (T022): the selected reference clip + actions to add
 * (upload a file / record from the mic), preview, and delete voices. Uploads
 * and recordings are re-encoded to 16-bit mono WAV in the browser so the
 * server always receives one format. */
function buildVoiceControl(value) {
  const wrap = document.createElement("div");
  wrap.className = "svoice";
  const sel = document.createElement("select");
  const opts = [...settings.voices];
  if (!opts.includes(value)) opts.unshift(value);
  for (const o of opts) {
    const opt = document.createElement("option");
    opt.value = o;
    opt.textContent = o;
    if (o === value) opt.selected = true;
    sel.appendChild(opt);
  }
  wrap.appendChild(sel);

  const status = document.createElement("span");
  status.className = "svoice-status";
  wrap.appendChild(status);
  const setStatus = (text, isErr) => {
    status.textContent = text || "";
    status.className = "svoice-status" + (isErr ? " err" : "");
  };

  const actions = document.createElement("div");
  actions.className = "svoice-actions";
  const mkBtn = (label, title) => {
    const b = document.createElement("button");
    b.className = "ghost tiny";
    b.textContent = label;
    b.title = title;
    actions.appendChild(b);
    return b;
  };
  const file = document.createElement("input");
  file.type = "file";
  file.accept = "audio/*";
  file.hidden = true;
  file.addEventListener("change", () => {
    const f = file.files[0];
    if (f) uploadVoice(f, f.name.replace(/\.[^.]+$/, ""));
    file.value = "";
  });
  const bUpload = mkBtn("upload", "Add a voice: choose an audio file (3-30 s of clear speech)");
  bUpload.addEventListener("click", () => file.click());
  const bRecord = mkBtn("record", "Add a voice: record from the microphone (speak for 3-30 s)");
  bRecord.addEventListener("click", () => toggleVoiceRecord(bRecord, setStatus));
  const bPreview = mkBtn("preview", "Play the selected reference clip");
  bPreview.addEventListener("click", () => previewVoice(sel.value, setStatus));
  const bDelete = mkBtn("delete", "Delete the selected reference clip");
  bDelete.addEventListener("click", () => deleteVoice(sel.value));
  wrap.appendChild(actions);
  return wrap;
}

let voiceRec = null; // { rec, stream, chunks, btn, status, setStatus, t0 }

async function toggleVoiceRecord(btn, setStatus) {
  if (voiceRec) {
    voiceRec.rec.stop();
    return;
  }
  let stream;
  try {
    stream = await navigator.mediaDevices.getUserMedia({ audio: true });
  } catch (err) {
    setStatus(`microphone: ${err.message || err}`, true);
    return;
  }
  const rec = new MediaRecorder(stream);
  const chunks = [];
  rec.ondataavailable = (e) => {
    if (e.data.size) chunks.push(e.data);
  };
  const t0 = Date.now();
  voiceRec = { rec, stream, chunks, btn, setStatus, t0 };
  rec.onstop = () => {
    const v = voiceRec;
    voiceRec = null;
    v.btn.textContent = "record";
    v.stream.getTracks().forEach((t) => t.stop());
    const secs = (Date.now() - v.t0) / 1000;
    if (secs < 3) {
      v.setStatus(`recording was ${secs.toFixed(1)}s — speak for 3-30 s`, true);
      return;
    }
    const blob = new Blob(chunks, { type: rec.mimeType || "audio/webm" });
    toWavBlob(blob)
      .then((wav) => uploadVoice(wav, "recorded"))
      .catch((err) => v.setStatus(`could not decode recording: ${err.message}`, true));
  };
  rec.start(250);
  btn.textContent = "stop";
  setStatus("recording… speak clearly, then stop");
}

async function toWavBlob(blob) {
  const buf = await blob.arrayBuffer();
  const Ctx = window.AudioContext || window.webkitAudioContext;
  const ctx = new Ctx();
  let audio;
  try {
    audio = await ctx.decodeAudioData(buf);
  } finally {
    ctx.close();
  }
  const chs = [];
  for (let c = 0; c < audio.numberOfChannels; c++) chs.push(audio.getChannelData(c));
  const n = audio.length;
  const mono = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    let s = 0;
    for (const ch of chs) s += ch[i];
    mono[i] = s / chs.length;
  }
  return encodeWav(mono, audio.sampleRate);
}

function encodeWav(f32, sr) {
  const n = f32.length;
  const buf = new ArrayBuffer(44 + n * 2);
  const v = new DataView(buf);
  const ws = (o, s) => {
    for (let i = 0; i < s.length; i++) v.setUint8(o + i, s.charCodeAt(i));
  };
  ws(0, "RIFF");
  v.setUint32(4, 36 + n * 2, true);
  ws(8, "WAVE");
  ws(12, "fmt ");
  v.setUint32(16, 16, true);
  v.setUint16(20, 1, true);
  v.setUint16(22, 1, true);
  v.setUint32(24, sr, true);
  v.setUint32(28, sr * 2, true);
  v.setUint16(32, 2, true);
  v.setUint16(34, 16, true);
  ws(36, "data");
  v.setUint32(40, n * 2, true);
  let o = 44;
  for (let i = 0; i < n; i++) {
    const s = Math.max(-1, Math.min(1, f32[i]));
    v.setInt16(o, s < 0 ? s * 0x8000 : s * 0x7fff, true);
    o += 2;
  }
  return new Blob([buf], { type: "audio/wav" });
}

async function uploadVoice(blob, name) {
  flashSettingsMsg(`saving voice "${name || "voice"}"…`);
  try {
    const wav = await toWavBlob(blob);
    const fd = new FormData();
    fd.append("file", wav, `${name || "voice"}.wav`);
    fd.append("name", name || "voice");
    const res = await fetch("/api/voice", { method: "POST", body: fd });
    const j = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(j.detail || `HTTP ${res.status}`);
    flashSettingsMsg(`voice "${j.voice}" saved — cloning in the background`);
    openSettings(); // re-render: the voice list + active voice changed
  } catch (err) {
    flashSettingsMsg(`voice upload failed: ${err.message}`, true);
  }
}

function previewVoice(name, setStatus) {
  if (!name) return;
  new Audio(`/api/voices/${encodeURIComponent(name)}`).play().catch((e) => {
    (setStatus || flashSettingsMsg)(`preview: ${e.message}`, true);
  });
}

async function deleteVoice(name) {
  if (!name || !confirm(`delete voice "${name}"?`)) return;
  const res = await fetch(`/api/voices/${encodeURIComponent(name)}`, { method: "DELETE" });
  const j = await res.json().catch(() => ({}));
  if (!res.ok) {
    flashSettingsMsg(j.detail || `HTTP ${res.status}`, true);
    return;
  }
  flashSettingsMsg(`voice "${name}" deleted`);
  openSettings();
}

function readSettingRow(row) {
  const t = row.dataset.type;
  if (t === "voices") return row.querySelector("select").value;
  const c = row.querySelector("input, select, textarea");
  if (t === "int") return Math.round(Number(c.value));
  if (t === "float") return Number(c.value);
  if (t === "bool") return c.checked;
  if (t === "str[]") return c.value.split("\n").map((s) => s.trim()).filter(Boolean);
  return c.value;
}

async function saveSettings() {
  if (!settings.schema) return;
  const values = {};
  for (const [sec, spec] of Object.entries(settings.schema)) {
    values[sec] = {};
    for (const key of Object.keys(spec.keys)) {
      const row = el.settingsBody.querySelector(`.srow[data-sec="${sec}"][data-key="${key}"]`);
      values[sec][key] = readSettingRow(row);
    }
  }
  el.btnSettingsSave.disabled = true;
  try {
    const res = await fetch("/api/config", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ values }),
    });
    const j = await res.json().catch(() => ({}));
    if (!res.ok) {
      const detail = Array.isArray(j.detail) ? j.detail.join("; ") : j.detail || `HTTP ${res.status}`;
      throw new Error(detail);
    }
    settings.values = j.values;
    flashSettingsMsg("saved — written to vivo.toml");
  } catch (err) {
    flashSettingsMsg(`save failed: ${err.message}`, true);
  } finally {
    el.btnSettingsSave.disabled = false;
  }
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
el.btnDream.addEventListener("click", triggerDream);

el.btnClear.addEventListener("click", () => {
  el.transcript.innerHTML = "";
  S.currentVivoEntry = null;
});

el.btnNewSession.addEventListener("click", () => sendJson({ type: "session" }));
el.btnRenameSession.addEventListener("click", renameSelectedSession);
el.btnDeleteSession.addEventListener("click", deleteSelectedSession);

el.sessionSelect.addEventListener("change", () => {
  if (el.sessionSelect.value) sendJson({ type: "session", id: el.sessionSelect.value });
});

if (el.motionPreset) {
  el.motionPreset.addEventListener("change", () => applyMotionPreset(el.motionPreset.value));
}

el.btnSettings.addEventListener("click", openSettings);
el.btnSettingsSave.addEventListener("click", saveSettings);
const closeSettings = () => el.settingsDlg.close();
el.btnSettingsClose.addEventListener("click", closeSettings);
el.btnSettingsX.addEventListener("click", closeSettings);
el.settingsDlg.addEventListener("click", (ev) => {
  if (ev.target === el.settingsDlg) closeSettings(); // backdrop click
});

window.addEventListener("resize", resizeCanvas);
window.addEventListener("load", () => {
  initMotionControls();
  resizeCanvas();
  loadSessions().then(connectWS); // bind a session before opening the socket
  requestAnimationFrame(frame);
  if (!window.isSecureContext || !navigator.mediaDevices) {
    el.btnStart.disabled = true;
    el.hint.innerHTML = "mic blocked &mdash; not a secure context (see sidebar)";
    addEntry("error", micUnavailableMsg());
  }
});
