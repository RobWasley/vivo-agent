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
  btnSettings: $("btn-settings"),
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
  if (k.type === "choices" || k.type === "voices") {
    const sel = document.createElement("select");
    const opts = k.type === "voices" ? [...settings.voices] : [...k.choices];
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

function readSettingRow(row) {
  const t = row.dataset.type;
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

el.btnClear.addEventListener("click", () => {
  el.transcript.innerHTML = "";
  S.currentVivoEntry = null;
});

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
  resizeCanvas();
  connectWS();
  requestAnimationFrame(frame);
  if (!window.isSecureContext || !navigator.mediaDevices) {
    el.btnStart.disabled = true;
    el.hint.innerHTML = "mic blocked &mdash; not a secure context (see sidebar)";
    addEntry("error", micUnavailableMsg());
  }
});
