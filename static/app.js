/* vivo UI: mic -> WS (16 kHz int16 PCM) ; WS TTS (48 kHz int16 PCM by default) -> speaker.
 *
 * Protocol (see app/pipeline.py):
  *   client -> server : binary 16 kHz int16 PCM,
  *                      {"type":"barge_in"|"flush"|"wake"|"ping"|{"type":"session"[,"id":str]}}
  *                      {"type":"text","message":str,"speech":bool}
  *                      (typed input: no VAD/STT/wake gate; speech=true -> TTS
  *                       reply, speech=false -> text-only reply)
 *   server -> client : JSON start|end|transcript|agent_text|tool|barge_ack|reply_done|error|session|config|wake
 *                      + binary int16 mono TTS chunks (one per sentence; sample rate via config.audio.tts_sample_rate)
 *                      agent_text deltas may carry start (first delta of a new
 *                      message -> new transcript entry + caption reset) and
 *                      filler (thinking filler phrase -> muted transcript line)
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
  caption: $("caption"),
  meterFill: $("meter-fill"),
  transcript: $("transcript"),
  transcriptToggle: $("transcript-toggle"),
  transcriptPanel: $("transcript-panel"),
  transcriptClose: $("transcript-close"),
  textInput: $("text-input"),
  btnSendText: $("btn-send-text"),
  btnSendSpeech: $("btn-send-speech"),
  sidenav: $("sidenav"),
  navToggle: $("nav-toggle"),
  btnStart: $("btn-start"),
  btnBarage: $("btn-barage"),
  btnFlush: $("btn-flush"),
  btnWake: $("btn-wake"),
  btnDream: $("btn-dream"),
  btnClear: $("btn-clear"),
  sessionSelect: $("session-select"),
  btnNewSession: $("btn-new-session"),
  btnRenameSession: $("btn-rename-session"),
  btnDeleteSession: $("btn-delete-session"),
  btnSettings: $("btn-settings"),
  btnSkills: $("btn-skills"),
  btnMemory: $("btn-memory"),
  motionPreset: $("motion-preset"),
  settingsDlg: $("settings"),
  settingsBody: $("settings-body"),
  settingsMsg: $("settings-msg"),
  btnSettingsSave: $("btn-settings-save"),
  btnSettingsClose: $("btn-settings-close"),
  btnSettingsX: $("btn-settings-x"),
  skillsDlg: $("skills"),
  skillsList: $("skills-list-items"),
  skillsMsg: $("skills-msg"),
  skillName: $("skill-name"),
  skillDescription: $("skill-description"),
  skillInstructions: $("skill-instructions"),
  btnSkillNew: $("btn-skill-new"),
  btnSkillSave: $("btn-skill-save"),
  btnSkillDelete: $("btn-skill-delete"),
  btnSkillsClose: $("btn-skills-close"),
  btnSkillsX: $("btn-skills-x"),
  memoryDlg: $("memory"),
  memoryList: $("memory-list-items"),
  memoryMsg: $("memory-msg"),
  memoryText: $("memory-text"),
  memoryCore: $("memory-core"),
  btnMemoryNew: $("btn-memory-new"),
  btnMemorySave: $("btn-memory-save"),
  btnMemoryDelete: $("btn-memory-delete"),
  btnMemoryDream: $("btn-memory-dream"),
  btnMemoryClose: $("btn-memory-close"),
  btnMemoryX: $("btn-memory-x"),
  hint: $("hint"),
  telUplink: $("tel-uplink"),
  telPipeline: $("tel-pipeline"),
  telMic: $("tel-mic"),
  telWake: $("tel-wake"),
  telSession: $("tel-session"),
  outputVolume: $("output-volume"),
  telVolume: $("tel-volume"),
};

const DEFAULT_OUTPUT_VOLUME = 0.7;
const OUTPUT_VOLUME_KEY = "vivo.output-volume";

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
  captionText: "",
  captionTimer: null,
  captionLingerMs: 8000, // overridden by the server's `config` message (vivo.toml [ui])
  pingTimer: null,
  bargeCfg: { ...BARGE_DEFAULTS }, // overridden by the server's `config` message
  bargePendingSince: null, // timestamp mic level started sustaining above threshold
  bargeCooldownUntil: 0, // suppress auto-barge re-triggers until this time
  dropAudio: false, // ignore in-flight TTS frames after a barge until the next `end`
  ttsSampleRate: DEFAULT_PLAY_RATE,
  outputVolume: DEFAULT_OUTPUT_VOLUME,
  dreaming: false,
  wakeEnabled: false, // wake phrase configured on the server (T024)
  wakeActive: false, // wake-phrase session currently awake
  wakePhrase: "", // the configured phrase, for the UI
  motionPreset: "balanced",
  prefersReducedMotion: false,
  hydrateToken: 0,
  sessionsById: {},
};

try { S.sessionId = localStorage.getItem("vivo.session"); } catch (_) { S.sessionId = null; }
try {
  const savedVolume = localStorage.getItem(OUTPUT_VOLUME_KEY);
  if (savedVolume !== null) {
    const volume = Number(savedVolume);
    if (Number.isFinite(volume) && volume >= 0 && volume <= 1) S.outputVolume = volume;
  }
} catch (_) {}

function setOutputVolume(value) {
  const volume = Math.max(0, Math.min(1, Number(value)));
  S.outputVolume = Number.isFinite(volume) ? volume : DEFAULT_OUTPUT_VOLUME;
  if (S.play) S.play.gain.gain.value = S.outputVolume;
  const percent = Math.round(S.outputVolume * 100);
  el.outputVolume.value = String(percent);
  el.telVolume.textContent = `${percent}%`;
  try { localStorage.setItem(OUTPUT_VOLUME_KEY, String(S.outputVolume)); } catch (_) {}
}

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
  if (s === "listening" || s === "thinking") {
    S.currentVivoEntry = null;
    S.captionText = "";
  }
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
  el.btnStart.classList.add("active");
  el.btnStart.title = "Stop microphone";
  el.btnStart.setAttribute("aria-label", "Stop microphone");
  const lbl = el.btnStart.querySelector(".nav-label");
  if (lbl) lbl.textContent = "Stop";
  updateWakeUi();
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
  el.btnStart.classList.remove("active");
  el.btnStart.title = "Start microphone";
  el.btnStart.setAttribute("aria-label", "Start microphone");
  const lbl = el.btnStart.querySelector(".nav-label");
  if (lbl) lbl.textContent = "Start";
  updateHint();
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
  gain.gain.value = S.outputVolume;
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
      if (m.ui) {
        const linger = Number(m.ui.caption_linger_s);
        if (Number.isFinite(linger) && linger > 0) S.captionLingerMs = linger * 1000;
      }
      if (m.wake) {
        S.wakePhrase = String(m.wake.phrase ?? "");
        S.wakeEnabled = S.wakePhrase.trim() !== "";
      }
      updateWakeUi();
      return; // config never touches the pipeline UI
    case "wake":
      S.wakeActive = !!m.active;
      updateWakeUi();
      return; // wake state never touches the pipeline UI
    case "dream":
      S.dreaming = !!m.active;
      updateStatus();
      return;
    case "start":
      if (S.wakeEnabled && !S.wakeActive) break; // dropped utterance while asleep
      setPipeline("listening");
      break;
    case "end":
      S.dropAudio = false; // a fresh reply: its audio is welcome again
      if (S.wakeEnabled && !S.wakeActive) break; // dropped utterance while asleep
      setPipeline("thinking");
      break;
    case "transcript":
      addEntry("you", m.text);
      setPipeline("thinking");
      break;
    case "agent_text":
      appendAgent(m.delta, !!m.start, !!m.filler);
      break;
    case "reminder":
      const reminderText = String(m.text || "Reminder");
      addEntry("hint", `⏰ reminder: ${escapeHtml(reminderText)}`);
      return;
    case "tool":
      addToolEntry(m.name, m.result, m.status, m.duration_ms);
      break;
    case "barge_ack":
      S.dropAudio = true;
      stopPlayback(); // belt-and-braces: barge may not have come from our button
      hideCaption();
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
      updateSessionTelemetry(m.id);
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

function tsNow() {
  const d = new Date();
  const p = (n) => String(n).padStart(2, "0");
  return `${p(d.getHours())}:${p(d.getMinutes())}:${p(d.getSeconds())}`;
}

function addEntry(cls, html, time) {
  const d = document.createElement("div");
  d.className = `entry entry-${cls}`;
  d.innerHTML = `<span class="t">${time || tsNow()}</span><span class="who">${cls}</span><p></p>`;
  d.querySelector("p").innerHTML = html;
  el.transcript.appendChild(d);
  scrollTranscript();
  return d;
}

function addToolEntry(name, result, status = "ok", durationMs = null) {
  const d = document.createElement("details");
  d.className = "entry entry-tool entry-tool-details";

  const summary = document.createElement("summary");
  summary.className = "tool-summary";

  const ts = document.createElement("span");
  ts.className = "t";
  ts.textContent = tsNow();
  summary.appendChild(ts);

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
  const timing = Number.isFinite(Number(durationMs)) ? ` ${Math.round(Number(durationMs))}ms` : "";
  toolHint.textContent = `${status === "error" ? "failed" : "done"}${timing} · click to expand`;
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

function appendAgent(delta, start, filler) {
  let entry = S.currentVivoEntry;
  if (start || !entry) {
    entry = document.createElement("div");
    entry.className = "entry entry-vivo" + (filler ? " entry-filler" : "");
    entry.innerHTML = `<span class="t">${tsNow()}</span><span class="who">vivo</span><p></p>`;
    el.transcript.appendChild(entry);
    S.currentVivoEntry = entry;
    S.captionText = ""; // the caption shows only the latest message
  }
  entry.querySelector("p").textContent += delta;
  S.captionText += delta;
  el.caption.textContent = S.captionText;
  el.caption.classList.add("show");
  if (S.captionTimer) clearTimeout(S.captionTimer);
  S.captionTimer = setTimeout(() => el.caption.classList.remove("show"), S.captionLingerMs);
  scrollTranscript();
}

function hideCaption() {
  if (S.captionTimer) clearTimeout(S.captionTimer);
  S.captionTimer = null;
  S.captionText = "";
  el.caption.classList.remove("show");
}

function scrollTranscript() {
  el.transcript.scrollTop = el.transcript.scrollHeight;
  if (!document.body.classList.contains("transcript-open")) {
    el.transcriptToggle.classList.add("has-new");
  }
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

  // the server stores no per-turn timestamps: mark hydrated turns accordingly
  if (j.summary) {
    addEntry("hint", `Earlier summary: ${escapeHtml(String(j.summary))}`, "&middot;&middot;&middot;");
  }
  for (const turn of j.turns || []) {
    if (turn.user) addEntry("you", escapeHtml(String(turn.user)), "&middot;&middot;&middot;");
    if (turn.assistant) addEntry("vivo", escapeHtml(String(turn.assistant)), "&middot;&middot;&middot;");
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

function updateSessionTelemetry(id) {
  if (!id) {
    el.telSession.textContent = "—";
    return;
  }
  const meta = S.sessionsById[id];
  const name = meta && (meta.name || "").trim();
  el.telSession.textContent = name || formatSessionId(id);
  el.telSession.title = el.telSession.textContent;
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
  updateSessionTelemetry(S.sessionId);
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
  let state;
  if (S.wsState === "closed") state = "closed";
  else if (S.wsState === "connecting") state = "connecting";
  else if (S.dreaming) state = "dreaming";
  else if (S.play && S.play.sources.size > 0) state = "speaking";
  else if (S.pipeline === "speaking") state = "speaking";
  else if (S.pipeline === "listening") state = "listening";
  else if (S.pipeline === "thinking") state = "thinking";
  else if (S.micActive && S.wakeEnabled && !S.wakeActive) state = "asleep";
  else state = "ready";

  el.telPipeline.dataset.state = state;
  el.telPipeline.textContent = state;
  el.telUplink.dataset.state = S.wsState;
  el.telUplink.textContent = S.wsState;
}

function updateWakeUi() {
  el.btnWake.disabled = !(S.wakeEnabled && !S.wakeActive);
  el.telWake.textContent = S.wakeActive ? "awake" : S.wakeEnabled ? S.wakePhrase : "standby";
  el.telWake.title = el.telWake.textContent;
  updateHint();
  updateStatus();
}

function updateHint() {
  if (S.micActive && S.wakeEnabled && !S.wakeActive) {
    el.hint.innerHTML =
      `sleeping &middot; say &ldquo;${escapeHtml(S.wakePhrase)}&rdquo; to wake vivo`;
  } else if (S.micActive) {
    el.hint.innerHTML = "mic on &middot; talk to me";
  } else {
    el.hint.innerHTML = "mic off &middot; click <b>start</b> to allow the microphone";
  }
}

/* ---------------- the core ring ----------------
 * A neon ring that reacts to the mic and playback levels: a breathing main
 * ring with a rotating gap and leading-edge tip, three orbiting segments,
 * an outer tick ring, a counter-rotating inner dash, and a soft glowing core.
 * State-specific overlays: ripples (listening), orbiting dots (thinking),
 * radial wave (speaking), drifting particles (dreaming), dimmed breath
 * (asleep), dashed spinner (connecting), broken flicker (closed).
 */

const canvas = el.canvas;
const c2d = canvas.getContext("2d");
let t = 0;
let telTick = 0;

function resizeCanvas() {
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  canvas.width = Math.max(1, Math.floor(rect.width * dpr));
  canvas.height = Math.max(1, Math.floor(rect.height * dpr));
  draw(); // resizing clears the canvas: repaint now, so the frame this
  // resize lands in is painted with content (ResizeObserver fires after
  // the rAF draw but before paint)
}

function rgbOf(hex) {
  const h = hex.slice(1);
  return {
    r: parseInt(h.slice(0, 2), 16),
    g: parseInt(h.slice(2, 4), 16),
    b: parseInt(h.slice(4, 6), 16),
  };
}

const STATE_COLORS = {
  connecting: [rgbOf("#9db2cf"), rgbOf("#26344e")],
  closed: [rgbOf("#ff5566"), rgbOf("#3c0d14")],
  idle: [rgbOf("#38bdf8"), rgbOf("#0c3f56")],
  listening: [rgbOf("#34d399"), rgbOf("#0a4430")],
  thinking: [rgbOf("#60a5fa"), rgbOf("#173263")],
  dreaming: [rgbOf("#a78bfa"), rgbOf("#2c2154")],
  speaking: [rgbOf("#fbbf24"), rgbOf("#5a3c0c")],
  asleep: [rgbOf("#5b6b80"), rgbOf("#161d29")],
  mic_off: [rgbOf("#8a96a8"), rgbOf("#202836")],
};

const MOTION_PRESETS = {
  calm: { speed: 0.6, glow: 0.65, ripple: 0.55, orbit: 0.55 },
  balanced: { speed: 1, glow: 1, ripple: 1, orbit: 1 },
  expressive: { speed: 1.5, glow: 1.4, ripple: 1.5, orbit: 1.5 },
};

const MOTION = { ...MOTION_PRESETS.balanced };
const VISUAL_TRANSITION_MS = 260;
const visual = { from: "idle", to: "idle", startedAt: performance.now() };

function clamp01(v) {
  return Math.max(0, Math.min(1, v));
}

function lerp(a, b, t) {
  return a + (b - a) * t;
}

function mixRgb(a, b, t) {
  return { r: lerp(a.r, b.r, t), g: lerp(a.g, b.g, t), b: lerp(a.b, b.b, t) };
}

function rgba(c, a) {
  const clamped = a > 1 ? 1 : a;
  if (clamped <= 0) return "rgba(0, 0, 0, 0)";
  return `rgba(${Math.round(c.r)}, ${Math.round(c.g)}, ${Math.round(c.b)}, ${clamped})`;
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
  if (!S.micActive) return "mic_off";
  if (S.dreaming) return "dreaming";
  if (S.play && S.play.sources.size > 0) return "speaking";
  if (S.pipeline === "idle" && S.micActive && S.wakeEnabled && !S.wakeActive) return "asleep";
  return S.pipeline === "idle" ? "idle" : S.pipeline;
}

function smoothLevel(prev, target) {
  const coef = target > prev ? 0.5 : 0.12;
  return prev + (target - prev) * coef;
}

/* ---------------- draw primitives ---------------- */

function ringArc(cx, cy, r, a0, a1, color, width, alpha, glowR = 0) {
  if (alpha <= 0.004) return;
  c2d.save();
  c2d.globalAlpha = Math.min(1, alpha);
  c2d.strokeStyle = color;
  c2d.lineWidth = width;
  c2d.lineCap = "round";
  if (glowR > 0) { c2d.shadowColor = color; c2d.shadowBlur = glowR; }
  c2d.beginPath();
  c2d.arc(cx, cy, r, a0, a1);
  c2d.stroke();
  c2d.restore();
}

function glowDot(x, y, r, color, alpha, glowR = 0) {
  if (alpha <= 0.004 || r <= 0) return;
  c2d.save();
  c2d.globalAlpha = Math.min(1, alpha);
  c2d.fillStyle = color;
  if (glowR > 0) { c2d.shadowColor = color; c2d.shadowBlur = glowR; }
  c2d.beginPath();
  c2d.arc(x, y, r, 0, Math.PI * 2);
  c2d.fill();
  c2d.restore();
}

/* ---------------- base stack ---------------- */

function drawTicks(cx, cy, R, time, color, weight, level) {
  if (weight < 0.02) return;
  const rot = time * 0.12 * MOTION.orbit;
  const n = 72;
  const r0 = R * 1.24;
  const baseA = (0.09 + 0.28 * level) * weight;
  c2d.save();
  c2d.strokeStyle = rgba(color, 1);
  c2d.lineCap = "round";
  for (let i = 0; i < n; i++) {
    const major = i % 6 === 0;
    const a = rot + (i / n) * Math.PI * 2;
    const len = major ? R * 0.075 : R * 0.032;
    c2d.globalAlpha = major ? Math.min(1, baseA * 1.6) : baseA;
    c2d.lineWidth = major ? 2 : 1;
    c2d.beginPath();
    c2d.moveTo(cx + Math.cos(a) * r0, cy + Math.sin(a) * r0);
    c2d.lineTo(cx + Math.cos(a) * (r0 + len), cy + Math.sin(a) * (r0 + len));
    c2d.stroke();
  }
  c2d.restore();
}

function drawOrbit(cx, cy, R, time, color, weight, level) {
  if (weight < 0.02) return;
  const r = R * 1.1;
  const rot = time * 0.55 * MOTION.orbit;
  const span = 1.05 + 0.3 * level;
  c2d.save();
  c2d.lineCap = "round";
  c2d.strokeStyle = rgba(color, 1);
  c2d.lineWidth = Math.max(1.5, R * 0.014);
  c2d.shadowColor = rgba(color, 1);
  for (let i = 0; i < 3; i++) {
    const a0 = rot + i * (Math.PI * 2 / 3);
    c2d.globalAlpha = (0.26 + 0.34 * level) * weight;
    c2d.shadowBlur = R * 0.1 * MOTION.glow * (0.4 + level);
    c2d.beginPath();
    c2d.arc(cx, cy, r, a0, a0 + span);
    c2d.stroke();
  }
  c2d.restore();
}

function drawMainRing(cx, cy, R, time, color, weight, level, speakingW) {
  if (weight < 0.02) return;
  const breathe = 0.5 + 0.5 * Math.sin(time * 1.3);
  const gap = Math.max(0.18, 0.38 + 0.26 * breathe - 0.3 * speakingW * (0.4 + 0.6 * level));
  const start = -time * 0.25 * MOTION.orbit;
  const c = rgba(color, 1);

  c2d.save();
  c2d.globalAlpha = 0.1 * weight;
  c2d.strokeStyle = c;
  c2d.lineWidth = Math.max(2, R * 0.03);
  c2d.beginPath();
  c2d.arc(cx, cy, R, 0, Math.PI * 2);
  c2d.stroke();
  c2d.restore();

  ringArc(cx, cy, R, start, start + Math.PI * 2 - gap, c, Math.max(2, R * 0.03), 0.9 * weight,
    R * 0.2 * MOTION.glow * (0.5 + level));

  const tipA = start + Math.PI * 2 - gap;
  glowDot(cx + Math.cos(tipA) * R, cy + Math.sin(tipA) * R, Math.max(2, R * 0.026), c, weight,
    R * 0.14 * MOTION.glow);
}

function drawInnerDash(cx, cy, R, time, color, weight) {
  if (weight < 0.02) return;
  c2d.save();
  c2d.globalAlpha = 0.22 * weight;
  c2d.strokeStyle = rgba(color, 1);
  c2d.lineWidth = Math.max(1, R * 0.008);
  c2d.setLineDash([R * 0.05, R * 0.11]);
  c2d.lineDashOffset = -time * R * 0.35 * MOTION.orbit;
  c2d.beginPath();
  c2d.arc(cx, cy, R * 0.86, 0, Math.PI * 2);
  c2d.stroke();
  c2d.restore();
}

function drawCore(cx, cy, R, inner, outer, level, weight, time) {
  if (weight <= 0.01) return;
  const r = R * (0.56 + 0.09 * level);
  const pulse = 0.5 + 0.5 * Math.sin(time * 1.15);
  const g = c2d.createRadialGradient(cx, cy, 0, cx, cy, r);
  g.addColorStop(0, rgba(mixRgb(inner, { r: 255, g: 255, b: 255 }, 0.45), (0.5 + 0.25 * pulse) * weight));
  g.addColorStop(0.55, rgba(inner, (0.2 + 0.2 * level) * weight));
  g.addColorStop(1, rgba(outer, 0));
  c2d.fillStyle = g;
  c2d.beginPath();
  c2d.arc(cx, cy, r, 0, Math.PI * 2);
  c2d.fill();
  glowDot(cx, cy, Math.max(1.5, R * 0.013),
    rgba(mixRgb(inner, { r: 255, g: 255, b: 255 }, 0.6), 0.9 * weight), R * 0.07 * MOTION.glow);
}

/* ---------------- state overlays ---------------- */

function drawConnecting(cx, cy, R, time, weight) {
  if (weight < 0.02) return;
  const c = STATE_COLORS.connecting[0];
  c2d.save();
  c2d.globalAlpha = 0.5 * weight;
  c2d.strokeStyle = rgba(c, 1);
  c2d.lineWidth = Math.max(1.5, R * 0.013);
  c2d.lineCap = "round";
  c2d.setLineDash([R * 0.16, R * 0.1]);
  c2d.lineDashOffset = -time * R * 0.5 * MOTION.speed;
  c2d.beginPath();
  c2d.arc(cx, cy, R * 1.18, 0, Math.PI * 2);
  c2d.stroke();
  c2d.restore();
}

function drawClosedFlicker(cx, cy, R, time, weight) {
  if (weight < 0.02) return;
  const c = STATE_COLORS.closed[0];
  const flick = 0.5 + 0.5 * Math.sin(time * 3.7);
  const a = (0.3 + 0.4 * flick) * weight;
  ringArc(cx, cy, R, 0.5 + 0.06 * flick, Math.PI - 0.4, rgba(c, 1), Math.max(1.5, R * 0.016), a, 0);
  ringArc(cx, cy, R, Math.PI + 0.5 - 0.06 * flick, Math.PI * 2 - 0.4, rgba(c, 1),
    Math.max(1.5, R * 0.016), a, 0);
}

function drawListeningRipples(cx, cy, R, time, weight) {
  if (weight < 0.02) return;
  const c = STATE_COLORS.listening[0];
  for (let i = 0; i < 3; i++) {
    const k = (time * 0.42 * MOTION.ripple + i / 3) % 1;
    const r = R * (1.02 + 0.5 * k);
    ringArc(cx, cy, r, 0, Math.PI * 2, rgba(c, 1), Math.max(1, R * 0.011),
      (1 - k) * 0.3 * weight, R * 0.08 * (1 - k) * MOTION.glow);
  }
}

function drawThinkingDots(cx, cy, R, time, weight) {
  if (weight < 0.02) return;
  const c = STATE_COLORS.thinking[0];
  const rot = time * 1.05 * MOTION.orbit;
  for (let i = 0; i < 3; i++) {
    const a = rot + i * (Math.PI * 2 / 3);
    const pulse = 0.55 + 0.45 * Math.sin(time * 2.3 + i * 2.1);
    glowDot(cx + Math.cos(a) * R * 1.1, cy + Math.sin(a) * R * 1.1,
      Math.max(2, R * 0.028) * pulse, rgba(c, 1), 0.8 * weight, R * 0.12 * MOTION.glow);
  }
  ringArc(cx, cy, R * 0.74, -rot * 1.4, -rot * 1.4 + 1.9, rgba(c, 1),
    Math.max(1.5, R * 0.013), 0.45 * weight, R * 0.08 * MOTION.glow);
  ringArc(cx, cy, R * 0.74, -rot * 1.4 + Math.PI, -rot * 1.4 + Math.PI + 1.9, rgba(c, 1),
    Math.max(1.5, R * 0.013), 0.45 * weight, R * 0.08 * MOTION.glow);
}

function drawSpeakingWave(cx, cy, R, time, weight, level) {
  if (weight < 0.02) return;
  const c = STATE_COLORS.speaking[0];
  const n = 28;
  const base = R * 0.7;
  c2d.save();
  c2d.lineCap = "round";
  c2d.strokeStyle = rgba(c, 1);
  c2d.lineWidth = Math.max(1.5, R * 0.015);
  for (let i = 0; i < n; i++) {
    const a = (i / n) * Math.PI * 2 + time * 0.3;
    const w = 0.5 + 0.5 * Math.sin(i * 0.9 + time * 6.5 * MOTION.speed);
    const len = R * (0.04 + (0.04 + 0.2 * level) * w);
    c2d.globalAlpha = (0.2 + 0.6 * w * (0.35 + 0.65 * level)) * weight;
    c2d.beginPath();
    c2d.moveTo(cx + Math.cos(a) * base, cy + Math.sin(a) * base);
    c2d.lineTo(cx + Math.cos(a) * (base + len), cy + Math.sin(a) * (base + len));
    c2d.stroke();
  }
  c2d.restore();
}

function drawDreaming(cx, cy, R, time, weight) {
  if (weight < 0.02) return;
  const c = STATE_COLORS.dreaming[0];
  const g = c2d.createRadialGradient(cx, cy, R * 0.5, cx, cy, R * 1.55);
  g.addColorStop(0, rgba(c, 0.09 * weight));
  g.addColorStop(1, rgba(c, 0));
  c2d.fillStyle = g;
  c2d.beginPath();
  c2d.arc(cx, cy, R * 1.55, 0, Math.PI * 2);
  c2d.fill();
  for (let i = 0; i < 5; i++) {
    const a = time * 0.24 * MOTION.orbit + i * 1.257;
    const rr = R * (1.02 + 0.16 * Math.sin(time * 0.7 + i * 2));
    const tw = 0.35 + 0.65 * (0.5 + 0.5 * Math.sin(time * 1.5 + i * 1.7));
    glowDot(cx + Math.cos(a) * rr, cy + Math.sin(a) * rr, Math.max(1.5, R * 0.018) * tw,
      rgba(c, 1), 0.7 * tw * weight, R * 0.1 * MOTION.glow);
  }
}

function drawAsleep(cx, cy, R, time, weight) {
  if (weight < 0.02) return;
  const c = STATE_COLORS.asleep[0];
  const pulse = 0.5 + 0.5 * Math.sin(time * 0.8);
  ringArc(cx, cy, R * (0.95 + 0.02 * pulse), 0, Math.PI * 2, rgba(c, 1),
    Math.max(1.5, R * 0.014), (0.14 + 0.1 * pulse) * weight, 0);
}

function drawMicOff(cx, cy, R, weight) {
  if (weight < 0.02) return;
  const c = STATE_COLORS.mic_off[0];
  ringArc(cx, cy, R * 0.96, Math.PI * 0.2, Math.PI * 1.8, rgba(c, 1),
    Math.max(1.5, R * 0.014), 0.28 * weight, 0);
  ringArc(cx, cy, R * 0.72, Math.PI * 1.12, Math.PI * 1.88, rgba(c, 1),
    Math.max(1.2, R * 0.01), 0.18 * weight, 0);
}

/* ---------------- frame loop ---------------- */

function frame() {
  t += (1 / 60) * MOTION.speed;
  S.micLevel = smoothLevel(S.micLevel, S.micLevelTarget);
  S.playLevel = smoothLevel(S.playLevel, readPlayLevel());

  draw();

  if (++telTick % 10 === 0) el.telMic.textContent = `${Math.round(S.micLevel * 100)}%`;

  el.meterFill.style.width = `${Math.round(S.micLevel * 100)}%`;
  requestAnimationFrame(frame);
}

function draw() {
  const now = performance.now();
  const w = canvas.width, h = canvas.height;
  c2d.clearRect(0, 0, w, h);
  const cx = w / 2, cy = h * 0.46;

  const level = Math.max(S.micLevel, S.playLevel);
  const coreState = currentDotState();
  updateVisualTarget(coreState, now);
  const progress = visualProgress(now);

  const f = STATE_COLORS[visual.from] || STATE_COLORS.idle;
  const to = STATE_COLORS[visual.to] || STATE_COLORS.idle;
  const inner = mixRgb(f[0], to[0], progress);
  const outer = mixRgb(f[1], to[1], progress);

  const speakingW = stateWeight("speaking", progress);
  const thinkingW = stateWeight("thinking", progress);
  const dreamingW = stateWeight("dreaming", progress);
  const asleepW = stateWeight("asleep", progress);
  const micOffW = stateWeight("mic_off", progress);
  const listeningW = stateWeight("listening", progress);
  const connectingW = stateWeight("connecting", progress);
  const closedW = stateWeight("closed", progress);

  // asleep: no level response, dim the stack, slow the rotation
  const inactiveW = Math.max(asleepW, micOffW);
  const reactLevel = level * (1 - inactiveW);
  const dim = 1 - inactiveW * 0.72;
  const flicker = closedW > 0.02
    ? (1 - closedW) + closedW * (0.55 + 0.45 * Math.sin(t * 3.7))
    : 1;

  const base = Math.min(w, h) * 0.27;
  const R = base * (1 + reactLevel * 0.05);

  drawTicks(cx, cy, R, t, inner, dim * flicker, reactLevel);
  drawOrbit(cx, cy, R, t, inner, dim * flicker, reactLevel);
  drawMainRing(cx, cy, R, t, inner, dim * flicker, reactLevel, speakingW);
  drawInnerDash(cx, cy, R, t, inner, dim * flicker);
  drawCore(cx, cy, R, inner, outer, reactLevel, dim * flicker, t);

  drawConnecting(cx, cy, R, t, connectingW);
  drawClosedFlicker(cx, cy, R, t, closedW);
  drawListeningRipples(cx, cy, R, t, listeningW);
  drawThinkingDots(cx, cy, R, t, thinkingW);
  drawSpeakingWave(cx, cy, R, t, speakingW, reactLevel);
  drawDreaming(cx, cy, R, t, dreamingW);
  drawAsleep(cx, cy, R, t, asleepW);
  drawMicOff(cx, cy, R, micOffW);
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
const SETTINGS_SECTION_KEY = "vivo.settings-section";
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
  let openSection = "voice";
  try { openSection = localStorage.getItem(SETTINGS_SECTION_KEY) || openSection; } catch (_) {}
  if (!Object.prototype.hasOwnProperty.call(j.schema, openSection)) openSection = "voice";
  for (const [sec, spec] of Object.entries(j.schema)) {
    const section = document.createElement("details");
    section.className = "ssection";
    section.dataset.section = sec;
    section.open = sec === openSection;
    const summary = document.createElement("summary");
    const title = document.createElement("span");
    title.textContent = spec.title;
    const count = document.createElement("span");
    count.className = "ssection-count";
    count.textContent = `${Object.keys(spec.keys).length} settings`;
    summary.append(title, count);
    section.appendChild(summary);
    const content = document.createElement("div");
    content.className = "ssection-content";
    for (const [key, k] of Object.entries(spec.keys)) {
      content.appendChild(buildSettingRow(sec, key, k));
    }
    section.appendChild(content);
    section.addEventListener("toggle", () => {
      if (!section.open) return;
      for (const other of el.settingsBody.querySelectorAll(".ssection[open]")) {
        if (other !== section) other.open = false;
      }
      try { localStorage.setItem(SETTINGS_SECTION_KEY, sec); } catch (_) {}
    });
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
    const val = document.createElement("input");
    val.className = "snumber";
    val.type = "number";
    val.min = k.min;
    val.max = k.max;
    val.step = k.step;
    val.value = value;
    range.addEventListener("input", () => { val.value = range.value; });
    val.addEventListener("input", () => {
      const numeric = Number(val.value);
      if (Number.isFinite(numeric) && numeric >= k.min && numeric <= k.max) range.value = String(numeric);
    });
    val.addEventListener("change", () => {
      const numeric = Number(val.value);
      const clamped = Number.isFinite(numeric) ? Math.max(k.min, Math.min(k.max, numeric)) : Number(range.value);
      range.value = String(clamped);
      val.value = range.value;
    });
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
  if (t === "int" || t === "float") {
    const value = Number(row.querySelector("input[type=number]").value);
    return t === "int" ? Math.round(value) : value;
  }
  const c = row.querySelector("input, select, textarea");
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

/* ---------------- skills ---------------- */

const skillsUi = { selected: "", items: [] };
let skillsMsgTimer = null;

function flashSkillsMsg(text, isError) {
  el.skillsMsg.textContent = text;
  el.skillsMsg.className = "settings-msg" + (isError ? " err" : " ok");
  if (skillsMsgTimer) clearTimeout(skillsMsgTimer);
  skillsMsgTimer = setTimeout(() => {
    el.skillsMsg.textContent = "";
    el.skillsMsg.className = "settings-msg";
  }, 3000);
}

function resetSkillEditor() {
  skillsUi.selected = "";
  el.skillName.value = "";
  el.skillDescription.value = "";
  el.skillInstructions.value = "";
  el.skillName.disabled = false;
  el.btnSkillDelete.disabled = true;
  renderSkillList();
  el.skillName.focus();
}

function renderSkillList() {
  el.skillsList.innerHTML = "";
  if (!skillsUi.items.length) {
    const empty = document.createElement("p");
    empty.className = "skills-empty";
    empty.textContent = "No skills yet.";
    el.skillsList.appendChild(empty);
    return;
  }
  for (const skill of skillsUi.items) {
    const item = document.createElement("button");
    item.type = "button";
    item.className = "skill-list-item" + (skill.name === skillsUi.selected ? " active" : "");
    const name = document.createElement("strong");
    name.textContent = skill.name;
    const description = document.createElement("span");
    description.textContent = skill.description;
    item.append(name, description);
    item.addEventListener("click", () => loadSkill(skill.name));
    el.skillsList.appendChild(item);
  }
}

async function refreshSkills(selected = skillsUi.selected) {
  const res = await fetch("/api/skills");
  const j = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(j.detail || `HTTP ${res.status}`);
  skillsUi.items = j.skills || [];
  if (!skillsUi.items.some((skill) => skill.name === selected)) selected = "";
  skillsUi.selected = selected;
  renderSkillList();
}

async function openSkills() {
  el.skillsDlg.showModal();
  resetSkillEditor();
  try {
    await refreshSkills();
  } catch (err) {
    flashSkillsMsg(`could not load skills: ${err.message}`, true);
  }
}

async function loadSkill(name) {
  try {
    const res = await fetch(`/api/skills/${encodeURIComponent(name)}`);
    const skill = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(skill.detail || `HTTP ${res.status}`);
    skillsUi.selected = skill.name;
    el.skillName.value = skill.name;
    el.skillDescription.value = skill.description;
    el.skillInstructions.value = skill.instructions;
    el.skillName.disabled = true;
    el.btnSkillDelete.disabled = false;
    renderSkillList();
  } catch (err) {
    flashSkillsMsg(`could not load skill: ${err.message}`, true);
  }
}

async function saveSkill() {
  if (!el.skillName.reportValidity() || !el.skillDescription.reportValidity() || !el.skillInstructions.reportValidity()) return;
  const name = el.skillName.value.trim();
  const body = {
    name,
    description: el.skillDescription.value.trim(),
    instructions: el.skillInstructions.value.trim(),
  };
  el.btnSkillSave.disabled = true;
  try {
    const res = await fetch(`/api/skills/${encodeURIComponent(name)}`, {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    const j = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(j.detail || `HTTP ${res.status}`);
    await refreshSkills(j.name);
    await loadSkill(j.name);
    flashSkillsMsg(`skill ${j.action}`);
  } catch (err) {
    flashSkillsMsg(`save failed: ${err.message}`, true);
  } finally {
    el.btnSkillSave.disabled = false;
  }
}

async function deleteSkill() {
  const name = skillsUi.selected;
  if (!name || !confirm(`delete skill "${name}"?`)) return;
  el.btnSkillDelete.disabled = true;
  try {
    const res = await fetch(`/api/skills/${encodeURIComponent(name)}`, { method: "DELETE" });
    const j = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(j.detail || `HTTP ${res.status}`);
    await refreshSkills();
    resetSkillEditor();
    flashSkillsMsg(`skill "${name}" deleted`);
  } catch (err) {
    flashSkillsMsg(`delete failed: ${err.message}`, true);
    el.btnSkillDelete.disabled = false;
  }
}

/* ---------------- memory ---------------- */

const memoryUi = { selected: "", facts: [] };
let memoryMsgTimer = null;

function flashMemoryMsg(text, isError) {
  el.memoryMsg.textContent = text;
  el.memoryMsg.className = "settings-msg" + (isError ? " err" : " ok");
  if (memoryMsgTimer) clearTimeout(memoryMsgTimer);
  memoryMsgTimer = setTimeout(() => {
    el.memoryMsg.textContent = "";
    el.memoryMsg.className = "settings-msg";
  }, 3000);
}

function resetMemoryEditor() {
  memoryUi.selected = "";
  el.memoryText.value = "";
  el.memoryCore.checked = false;
  el.btnMemoryDelete.disabled = true;
  renderMemoryList();
  el.memoryText.focus();
}

function renderMemoryList() {
  el.memoryList.innerHTML = "";
  if (!memoryUi.facts.length) {
    const empty = document.createElement("p");
    empty.className = "skills-empty";
    empty.textContent = "No saved facts.";
    el.memoryList.appendChild(empty);
    return;
  }
  for (const fact of memoryUi.facts) {
    const item = document.createElement("button");
    item.type = "button";
    item.className = "skill-list-item" + (fact.id === memoryUi.selected ? " active" : "");
    const text = document.createElement("strong");
    text.textContent = fact.text;
    const meta = document.createElement("span");
    meta.textContent = `${fact.date}${fact.core ? " · core" : " · archive"}`;
    item.append(text, meta);
    item.addEventListener("click", () => loadMemory(fact.id));
    el.memoryList.appendChild(item);
  }
}

async function refreshMemory(selected = memoryUi.selected) {
  const res = await fetch("/api/memory");
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.detail || `HTTP ${res.status}`);
  memoryUi.facts = data.facts || [];
  if (!memoryUi.facts.some((fact) => fact.id === selected)) selected = "";
  memoryUi.selected = selected;
  renderMemoryList();
}

async function openMemory() {
  el.memoryDlg.showModal();
  resetMemoryEditor();
  try {
    await refreshMemory();
  } catch (err) {
    flashMemoryMsg(`could not load memory: ${err.message}`, true);
  }
}

function loadMemory(id) {
  const fact = memoryUi.facts.find((item) => item.id === id);
  if (!fact) return;
  memoryUi.selected = fact.id;
  el.memoryText.value = fact.text;
  el.memoryCore.checked = !!fact.core;
  el.btnMemoryDelete.disabled = false;
  renderMemoryList();
}

async function saveMemory() {
  const text = el.memoryText.value.trim();
  if (!text) return el.memoryText.reportValidity();
  const body = { text, core: el.memoryCore.checked };
  const isNew = !memoryUi.selected;
  const endpoint = isNew ? "/api/memory" : `/api/memory/${encodeURIComponent(memoryUi.selected)}`;
  el.btnMemorySave.disabled = true;
  try {
    const res = await fetch(endpoint, {
      method: isNew ? "POST" : "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(data.detail || `HTTP ${res.status}`);
    await refreshMemory(data.fact.id);
    loadMemory(data.fact.id);
    flashMemoryMsg(isNew ? "fact saved" : "fact updated");
  } catch (err) {
    flashMemoryMsg(`save failed: ${err.message}`, true);
  } finally {
    el.btnMemorySave.disabled = false;
  }
}

async function deleteMemory() {
  const id = memoryUi.selected;
  if (!id || !confirm("delete this memory fact?")) return;
  el.btnMemoryDelete.disabled = true;
  try {
    const res = await fetch(`/api/memory/${encodeURIComponent(id)}`, { method: "DELETE" });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(data.detail || `HTTP ${res.status}`);
    await refreshMemory();
    resetMemoryEditor();
    flashMemoryMsg("fact deleted");
  } catch (err) {
    flashMemoryMsg(`delete failed: ${err.message}`, true);
    el.btnMemoryDelete.disabled = false;
  }
}

async function dreamMemory() {
  el.btnMemoryDream.disabled = true;
  try {
    const res = await fetch("/api/dream", { method: "POST" });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(data.detail || `HTTP ${res.status}`);
    await refreshMemory(memoryUi.selected);
    flashMemoryMsg(data.summary === "No important memory yet." ? "core memory cleared" : "core memory refreshed");
  } catch (err) {
    flashMemoryMsg(`dream failed: ${err.message}`, true);
  } finally {
    el.btnMemoryDream.disabled = false;
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

function sendText(speech) {
  const text = el.textInput.value.trim();
  if (!text || !S.ws || S.ws.readyState !== WebSocket.OPEN) return;
  sendJson({ type: "text", message: text, speech: speech });
  el.textInput.value = "";
}
el.btnSendText.addEventListener("click", () => sendText(false));
el.btnSendSpeech.addEventListener("click", () => sendText(true));
el.textInput.addEventListener("keydown", (ev) => {
  if (ev.key === "Enter") {
    ev.preventDefault();
    sendText(false);
  }
});
el.btnWake.addEventListener("click", () => sendJson({ type: "wake" }));
el.btnDream.addEventListener("click", triggerDream);
el.outputVolume.addEventListener("input", () => setOutputVolume(Number(el.outputVolume.value) / 100));

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

el.navToggle.addEventListener("click", () => {
  const expanded = el.sidenav.classList.toggle("expanded");
  document.body.classList.toggle("nav-expanded", expanded);
  el.navToggle.setAttribute("aria-expanded", expanded ? "true" : "false");
  el.navToggle.title = expanded ? "Collapse controls" : "Expand controls";
});

function setTranscriptOpen(open) {
  document.body.classList.toggle("transcript-open", open);
  el.transcriptToggle.setAttribute("aria-expanded", open ? "true" : "false");
  if (open) {
    el.transcriptToggle.classList.remove("has-new");
    scrollTranscript();
  }
}

el.transcriptToggle.addEventListener("click", () =>
  setTranscriptOpen(!document.body.classList.contains("transcript-open")));
el.transcriptClose.addEventListener("click", () => setTranscriptOpen(false));

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

el.btnSkills.addEventListener("click", openSkills);
el.btnSkillNew.addEventListener("click", resetSkillEditor);
el.btnSkillSave.addEventListener("click", saveSkill);
el.btnSkillDelete.addEventListener("click", deleteSkill);
const closeSkills = () => el.skillsDlg.close();
el.btnSkillsClose.addEventListener("click", closeSkills);
el.btnSkillsX.addEventListener("click", closeSkills);
el.skillsDlg.addEventListener("click", (ev) => {
  if (ev.target === el.skillsDlg) closeSkills();
});

el.btnMemory.addEventListener("click", openMemory);
el.btnMemoryNew.addEventListener("click", resetMemoryEditor);
el.btnMemorySave.addEventListener("click", saveMemory);
el.btnMemoryDelete.addEventListener("click", deleteMemory);
el.btnMemoryDream.addEventListener("click", dreamMemory);
const closeMemory = () => el.memoryDlg.close();
el.btnMemoryClose.addEventListener("click", closeMemory);
el.btnMemoryX.addEventListener("click", closeMemory);
el.memoryDlg.addEventListener("click", (ev) => {
  if (ev.target === el.memoryDlg) closeMemory();
});

window.addEventListener("resize", resizeCanvas);
new ResizeObserver(resizeCanvas).observe(el.canvas);
window.addEventListener("load", () => {
  setOutputVolume(S.outputVolume);
  initMotionControls();
  resizeCanvas();
  loadSessions().then(connectWS); // bind a session before opening the socket
  requestAnimationFrame(frame);
  if (!window.isSecureContext || !navigator.mediaDevices) {
    el.btnStart.disabled = true;
    el.hint.innerHTML = "mic blocked &mdash; not a secure context";
    addEntry("error", micUnavailableMsg());
  }
});
