/* Realtime Voice + Vision — Client
 *
 * Architecture matches Parlor: Silero VAD in browser, streaming PCM playback
 * via AudioContext, state machine with visual feedback, echo-suppressed
 * barge-in, and auto-reconnect.
 *
 * Server differences from Parlor: we use server-side ASR (faster-whisper)
 * instead of native audio understanding, and stream LLM tokens for live text.
 */

const $ = (id) => document.getElementById(id);

const video = $("video");
const cameraToggle = $("cameraToggle");
const messagesDiv = $("messages");
const statusPill = $("statusPill");
const stateDot = $("stateDot");
const stateText = $("stateText");
const viewportWrap = $("viewportWrap");
const modelLabel = $("modelLabel");
const textInput = $("textInput");
const sendBtn = $("sendBtn");
const waveformCanvas = $("waveform");
const waveformCtx = waveformCanvas.getContext("2d");

let ws, mediaStream, myvad;
let cameraEnabled = true;
let audioCtx, analyser, micSource;
let appState = "loading";

// Streaming PCM playback state
let streamSampleRate = 24000;
let streamNextTime = 0;
let streamSources = [];
let ignoreIncomingAudio = false;

// Barge-in echo suppression
let speakingStartedAt = 0;
const BARGE_IN_GRACE_MS = 800;

// Streaming assistant text
let assistantBubble = null;

// Waveform
const BAR_COUNT = 40;
const BAR_GAP = 3;
let ambientPhase = 0;
let waveformRAF;

// ─────────────────────────────────────────
// Waveform Visualizer (from Parlor)
// ─────────────────────────────────────────

function initWaveformCanvas() {
  const dpr = window.devicePixelRatio || 1;
  const rect = waveformCanvas.getBoundingClientRect();
  waveformCanvas.width = rect.width * dpr;
  waveformCanvas.height = rect.height * dpr;
  waveformCtx.scale(dpr, dpr);
}

function getStateColor() {
  const colors = {
    listening: "#4ade80",
    processing: "#f59e0b",
    speaking: "#818cf8",
    loading: "#3a3d46",
  };
  return colors[appState] || colors.loading;
}

function drawWaveform() {
  const w = waveformCanvas.getBoundingClientRect().width;
  const h = waveformCanvas.getBoundingClientRect().height;
  waveformCtx.clearRect(0, 0, w, h);

  const barW = (w - (BAR_COUNT - 1) * BAR_GAP) / BAR_COUNT;
  waveformCtx.fillStyle = getStateColor();

  let dataArray = null;
  if (analyser) {
    dataArray = new Uint8Array(analyser.frequencyBinCount);
    analyser.getByteFrequencyData(dataArray);
  }

  for (let i = 0; i < BAR_COUNT; i++) {
    let amp;
    if (dataArray) {
      const bin = Math.floor((i / BAR_COUNT) * dataArray.length * 0.6);
      amp = dataArray[bin] / 255;
    }
    if (!dataArray || amp < 0.02) {
      ambientPhase += 0.0001;
      const drift = Math.sin(ambientPhase * 3 + i * 0.4) * 0.5 + 0.5;
      amp = 0.03 + drift * 0.04;
    }
    const barH = Math.max(2, amp * (h - 4));
    const x = i * (barW + BAR_GAP);
    const y = (h - barH) / 2;
    waveformCtx.globalAlpha = 0.3 + amp * 0.7;
    waveformCtx.beginPath();
    const r = Math.min(barW / 2, barH / 2, 3);
    waveformCtx.roundRect(x, y, barW, barH, r);
    waveformCtx.fill();
  }
  waveformCtx.globalAlpha = 1;
  waveformRAF = requestAnimationFrame(drawWaveform);
}

function updateSpeakingGlow() {
  if (appState !== "speaking" || !analyser) return;
  const data = new Uint8Array(analyser.frequencyBinCount);
  analyser.getByteFrequencyData(data);
  let sum = 0;
  for (let i = 0; i < data.length; i++) sum += data[i];
  const avg = sum / data.length / 255;
  const intensity = 0.3 + avg * 0.7;
  const spread = 20 + avg * 60;
  const inner = 15 + avg * 25;
  const glow = viewportWrap.querySelector(".viewport-glow");
  if (glow) {
    glow.style.boxShadow = `0 0 ${spread}px ${spread * 0.4}px rgba(129,140,248,${intensity * 0.25})`;
  }
  viewportWrap.style.boxShadow =
    `inset 0 0 ${inner}px rgba(129,140,248,${intensity * 0.15}), 0 0 ${inner}px rgba(129,140,248,${intensity * 0.2})`;
  requestAnimationFrame(updateSpeakingGlow);
}

// ─────────────────────────────────────────
// State Machine
// ─────────────────────────────────────────

function setState(newState) {
  appState = newState;
  stateDot.className = `dot ${newState}`;
  const labels = {
    loading: "Loading...",
    listening: "Listening",
    processing: "Thinking...",
    speaking: "Speaking",
  };
  stateText.textContent = labels[newState] || newState;
  viewportWrap.className = `viewport-wrap ${newState}`;

  if (newState !== "speaking") {
    viewportWrap.style.boxShadow = "";
    const glow = viewportWrap.querySelector(".viewport-glow");
    if (glow) glow.style.boxShadow = "";
  }

  const stateVars = {
    listening: ["#4ade80", "rgba(74,222,128,0.12)"],
    processing: ["#f59e0b", "rgba(245,158,11,0.12)"],
    speaking: ["#818cf8", "rgba(129,140,248,0.12)"],
    loading: ["#3a3d46", "rgba(58,61,70,0.12)"],
  };
  const [g, gd] = stateVars[newState] || stateVars.loading;
  document.documentElement.style.setProperty("--glow", g);
  document.documentElement.style.setProperty("--glow-dim", gd);

  if (newState === "speaking") requestAnimationFrame(updateSpeakingGlow);

  // Raise VAD threshold during speaking to suppress echo
  if (myvad && typeof myvad.setOptions === "function") {
    myvad.setOptions({
      positiveSpeechThreshold: newState === "speaking" ? 0.92 : 0.5,
    });
  }

  // Connect/disconnect mic analyser
  if (newState === "listening" && mediaStream && audioCtx && analyser) {
    if (!micSource) {
      micSource = audioCtx.createMediaStreamSource(mediaStream);
    }
    try { micSource.connect(analyser); } catch (_) {}
  } else if (micSource && newState !== "listening") {
    try { micSource.disconnect(analyser); } catch (_) {}
  }
}

// ─────────────────────────────────────────
// Audio Utilities
// ─────────────────────────────────────────

function float32ToWavBase64(samples) {
  const buf = new ArrayBuffer(44 + samples.length * 2);
  const v = new DataView(buf);
  const w = (o, s) => {
    for (let i = 0; i < s.length; i++) v.setUint8(o + i, s.charCodeAt(i));
  };
  w(0, "RIFF");
  v.setUint32(4, 36 + samples.length * 2, true);
  w(8, "WAVE");
  w(12, "fmt ");
  v.setUint32(16, 16, true);
  v.setUint16(20, 1, true);
  v.setUint16(22, 1, true);
  v.setUint32(24, 16000, true);
  v.setUint32(28, 32000, true);
  v.setUint16(32, 2, true);
  v.setUint16(34, 16, true);
  w(36, "data");
  v.setUint32(40, samples.length * 2, true);
  for (let i = 0; i < samples.length; i++) {
    const s = Math.max(-1, Math.min(1, samples[i]));
    v.setInt16(44 + i * 2, s < 0 ? s * 0x8000 : s * 0x7fff, true);
  }
  const bytes = new Uint8Array(buf);
  let bin = "";
  for (let i = 0; i < bytes.length; i++) bin += String.fromCharCode(bytes[i]);
  return btoa(bin);
}

function captureFrame() {
  if (!cameraEnabled || !video.videoWidth) return null;
  const canvas = document.createElement("canvas");
  const scale = 320 / video.videoWidth;
  canvas.width = 320;
  canvas.height = video.videoHeight * scale;
  canvas.getContext("2d").drawImage(video, 0, 0, canvas.width, canvas.height);
  return canvas.toDataURL("image/jpeg", 0.7).split(",")[1];
}

function ensureAudioCtx() {
  if (!audioCtx) {
    audioCtx = new AudioContext();
    analyser = audioCtx.createAnalyser();
    analyser.fftSize = 256;
    analyser.smoothingTimeConstant = 0.75;
  }
}

// ─────────────────────────────────────────
// Streaming PCM Playback (from Parlor)
// ─────────────────────────────────────────

function stopPlayback() {
  for (const src of streamSources) {
    try { src.stop(); } catch (_) {}
  }
  streamSources = [];
  streamNextTime = 0;
}

function startStreamPlayback(sampleRate) {
  stopPlayback();
  ensureAudioCtx();
  if (audioCtx.state === "suspended") audioCtx.resume();
  streamSampleRate = sampleRate || 24000;
  streamNextTime = audioCtx.currentTime + 0.05;
  speakingStartedAt = Date.now();
  setState("speaking");
}

function queueAudioChunk(base64Pcm) {
  ensureAudioCtx();
  const bin = atob(base64Pcm);
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  const int16 = new Int16Array(bytes.buffer);
  const float32 = new Float32Array(int16.length);
  for (let i = 0; i < int16.length; i++) float32[i] = int16[i] / 32768;

  const audioBuffer = audioCtx.createBuffer(1, float32.length, streamSampleRate);
  audioBuffer.getChannelData(0).set(float32);

  const source = audioCtx.createBufferSource();
  source.buffer = audioBuffer;
  source.connect(audioCtx.destination);
  source.connect(analyser);

  const startAt = Math.max(streamNextTime, audioCtx.currentTime);
  source.start(startAt);
  streamNextTime = startAt + audioBuffer.duration;
  streamSources.push(source);

  source.onended = () => {
    const idx = streamSources.indexOf(source);
    if (idx !== -1) streamSources.splice(idx, 1);
    if (streamSources.length === 0 && appState === "speaking") {
      setState("listening");
      setStatus("connected", "Connected");
    }
  };
}

// ─────────────────────────────────────────
// VAD Handlers
// ─────────────────────────────────────────

function handleSpeechStart() {
  if (appState === "speaking") {
    if (Date.now() - speakingStartedAt < BARGE_IN_GRACE_MS) {
      console.log("Barge-in suppressed (echo grace period)");
      return;
    }
    stopPlayback();
    ignoreIncomingAudio = true;
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ type: "interrupt" }));
    }
    setState("listening");
    console.log("Barge-in: interrupted playback");
  }
}

function handleSpeechEnd(audio) {
  if (appState !== "listening") return;
  if (!ws || ws.readyState !== WebSocket.OPEN) return;

  const wavBase64 = float32ToWavBase64(audio);
  const imageBase64 = captureFrame();

  setState("processing");
  setStatus("processing", "Processing");
  addMessage("user", "\u00a0", imageBase64 ? "with camera" : "");

  const payload = { audio: wavBase64 };
  if (imageBase64) payload.image = imageBase64;
  ws.send(JSON.stringify(payload));
}

// ─────────────────────────────────────────
// WebSocket
// ─────────────────────────────────────────

function setStatus(cls, text) {
  statusPill.className = `status-pill ${cls}`;
  statusPill.textContent = text;
}

function connect() {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  ws = new WebSocket(`${proto}://${location.host}/ws`);

  ws.onopen = () => {
    setStatus("connected", "Connected");
    if (appState !== "loading") setState("listening");
  };

  ws.onclose = () => {
    setStatus("disconnected", "Disconnected");
    setTimeout(connect, 2000);
  };

  ws.onmessage = ({ data }) => {
    const msg = JSON.parse(data);

    if (msg.type === "ready") {
      modelLabel.textContent = msg.config?.model_name || "unknown";
      return;
    }

    if (msg.type === "status") {
      if (msg.phase === "transcribing") setState("processing");
      else if (msg.phase === "thinking") setState("processing");
      return;
    }

    if (msg.type === "text") {
      if (msg.transcription) {
        // Update the loading-dots placeholder with actual transcription
        const userMsgs = messagesDiv.querySelectorAll(".msg.user");
        const lastUser = userMsgs[userMsgs.length - 1];
        if (lastUser) {
          const meta = lastUser.querySelector(".meta");
          lastUser.innerHTML = `${msg.transcription}${meta ? meta.outerHTML : ""}`;
        }
      }
      if (msg.text && msg.llm_time !== undefined) {
        // Final assistant response with timing
        finalizeAssistant(msg.text, `LLM ${msg.llm_time}s`);
      }
      return;
    }

    if (msg.type === "assistant_token") {
      updateAssistantToken(msg.text);
      return;
    }

    if (msg.type === "audio_start") {
      if (ignoreIncomingAudio) return;
      startStreamPlayback(msg.sample_rate);
      return;
    }

    if (msg.type === "audio_chunk") {
      if (ignoreIncomingAudio) return;
      queueAudioChunk(msg.audio);
      return;
    }

    if (msg.type === "audio_end") {
      if (ignoreIncomingAudio) {
        ignoreIncomingAudio = false;
        stopPlayback();
        setState("listening");
        return;
      }
      if (msg.tts_time !== undefined) {
        const meta = messagesDiv.querySelector(".msg.assistant:last-child .meta");
        if (meta) meta.textContent += ` \u00b7 TTS ${msg.tts_time}s`;
      }
      return;
    }

    if (msg.type === "error") {
      addMessage("system", msg.message);
      if (appState === "processing") setState("listening");
      return;
    }
  };
}

// ─────────────────────────────────────────
// UI Helpers
// ─────────────────────────────────────────

function addMessage(role, text, meta) {
  const div = document.createElement("div");
  div.className = `msg ${role}`;
  if (role === "user" && !text.trim()) {
    div.innerHTML =
      '<span class="loading-dots"><span></span><span></span><span></span></span>';
  } else {
    const span = document.createElement("span");
    span.className = "msg-text";
    span.textContent = text;
    div.appendChild(span);
  }
  if (meta) {
    const m = document.createElement("div");
    m.className = "meta";
    m.textContent = meta;
    div.appendChild(m);
  }
  messagesDiv.appendChild(div);
  messagesDiv.scrollTop = messagesDiv.scrollHeight;
  return div;
}

function updateAssistantToken(text) {
  if (!assistantBubble) {
    assistantBubble = addMessage("assistant", "");
  }
  const span = assistantBubble.querySelector(".msg-text");
  if (span) {
    span.textContent += text;
  }
  messagesDiv.scrollTop = messagesDiv.scrollHeight;
}

function finalizeAssistant(text, meta) {
  if (!assistantBubble) {
    assistantBubble = addMessage("assistant", text, meta);
  } else {
    const span = assistantBubble.querySelector(".msg-text");
    if (span) span.textContent = text;
    if (meta) {
      const m = document.createElement("div");
      m.className = "meta";
      m.textContent = meta;
      assistantBubble.appendChild(m);
    }
  }
  assistantBubble = null;
}

// ─────────────────────────────────────────
// Camera
// ─────────────────────────────────────────

async function startCamera() {
  try {
    mediaStream = await navigator.mediaDevices.getUserMedia({
      video: { width: 640, height: 480, facingMode: "user" },
      audio: {
        echoCancellation: true,
        noiseSuppression: true,
        autoGainControl: true,
      },
    });
    video.srcObject = mediaStream;
    return;
  } catch (e) {
    console.warn("Video+audio failed:", e.message);
  }

  // Fallback: request separately
  const streams = await Promise.allSettled([
    navigator.mediaDevices.getUserMedia({
      video: { width: 640, height: 480, facingMode: "user" },
    }),
    navigator.mediaDevices.getUserMedia({
      audio: {
        echoCancellation: true,
        noiseSuppression: true,
        autoGainControl: true,
      },
    }),
  ]);
  mediaStream = new MediaStream();
  streams.forEach((r) => {
    if (r.status === "fulfilled")
      r.value.getTracks().forEach((t) => mediaStream.addTrack(t));
  });
  if (mediaStream.getVideoTracks().length) video.srcObject = mediaStream;
  if (!mediaStream.getAudioTracks().length) cameraEnabled = false;
}

cameraToggle.addEventListener("click", () => {
  cameraEnabled = !cameraEnabled;
  cameraToggle.classList.toggle("active", cameraEnabled);
  cameraToggle.textContent = cameraEnabled ? "Camera On" : "Camera Off";
  video.style.opacity = cameraEnabled ? 1 : 0.3;
});

// ─────────────────────────────────────────
// Text input (bypass ASR)
// ─────────────────────────────────────────

function sendTextMessage() {
  const text = textInput.value.trim();
  if (!text || !ws || ws.readyState !== WebSocket.OPEN) return;
  addMessage("user", text);
  assistantBubble = null;
  ws.send(JSON.stringify({ type: "text_input", text }));
  textInput.value = "";
  setState("processing");
  setStatus("processing", "Processing");
}

sendBtn.addEventListener("click", sendTextMessage);
textInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    sendTextMessage();
  }
});

// ─────────────────────────────────────────
// Init
// ─────────────────────────────────────────

async function init() {
  initWaveformCanvas();
  window.addEventListener("resize", initWaveformCanvas);

  await startCamera();
  connect();

  // Initialize Silero VAD with shared mic stream
  try {
    myvad = await vad.MicVAD.new({
      stream: mediaStream,
      positiveSpeechThreshold: 0.5,
      negativeSpeechThreshold: 0.25,
      redemptionMs: 600,
      minSpeechMs: 300,
      preSpeechPadMs: 300,
      onSpeechStart: handleSpeechStart,
      onSpeechEnd: handleSpeechEnd,
      onVADMisfire: () => console.log("VAD misfire (too short)"),
      onnxWASMBasePath: "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.22.0/dist/",
      baseAssetPath: "https://cdn.jsdelivr.net/npm/@ricky0123/vad-web@0.0.29/dist/",
    });
    myvad.start();
    console.log("Silero VAD initialized");
  } catch (err) {
    console.warn("Silero VAD failed, using energy fallback:", err.message);
    startEnergyVAD();
  }

  // Init AudioContext on first user gesture
  const initAudio = () => {
    ensureAudioCtx();
    if (audioCtx.state === "suspended") audioCtx.resume();
    document.removeEventListener("click", initAudio);
    document.removeEventListener("keydown", initAudio);
  };
  document.addEventListener("click", initAudio);
  document.addEventListener("keydown", initAudio);
  ensureAudioCtx();

  setState("listening");
  drawWaveform();
}

// ─────────────────────────────────────────
// Energy VAD fallback
// ─────────────────────────────────────────

function startEnergyVAD() {
  if (!mediaStream || !mediaStream.getAudioTracks().length) return;
  const ctx = new AudioContext();
  const source = ctx.createMediaStreamSource(mediaStream);
  const processor = ctx.createScriptProcessor(4096, 1, 1);
  source.connect(processor);
  processor.connect(ctx.destination);

  let active = false;
  let chunks = [];
  let preRoll = [];
  let silenceMs = 0;
  let speechMs = 0;
  let hotFrames = 0;
  const threshold = 0.01;
  const startFrames = 3;
  const minSpeechMs = 500;
  const maxPreRollSamples = Math.round(0.25 * 16000);

  function downsample(data, inputRate, targetRate) {
    if (inputRate === targetRate) return data;
    const ratio = inputRate / targetRate;
    const len = Math.round(data.length / ratio);
    const out = new Float32Array(len);
    for (let i = 0; i < len; i++) {
      const start = Math.round(i * ratio);
      const end = Math.min(Math.round((i + 1) * ratio), data.length);
      let sum = 0;
      for (let j = start; j < end; j++) sum += data[j];
      out[i] = sum / (end - start || 1);
    }
    return out;
  }

  processor.onaudioprocess = (e) => {
    const raw = new Float32Array(e.inputBuffer.getChannelData(0));
    const samples = downsample(raw, ctx.sampleRate, 16000);
    let rms = 0;
    for (let i = 0; i < samples.length; i++) rms += samples[i] * samples[i];
    rms = Math.sqrt(rms / samples.length);

    const chunkMs = (samples.length / 16000) * 1000;

    if (!active) {
      preRoll.push(samples);
      let total = preRoll.reduce((s, c) => s + c.length, 0);
      while (total > maxPreRollSamples && preRoll.length > 1) {
        total -= preRoll.shift().length;
      }
    }

    if (rms > threshold) {
      hotFrames++;
      silenceMs = 0;
      if (!active && hotFrames >= startFrames) {
        active = true;
        chunks = [...preRoll];
        preRoll = [];
        speechMs = 0;
        handleSpeechStart();
      }
    } else {
      hotFrames = 0;
      if (active) silenceMs += chunkMs;
    }

    if (active) {
      chunks.push(samples);
      speechMs += chunkMs;
    }

    if (active && silenceMs > 700) {
      const len = chunks.reduce((s, c) => s + c.length, 0);
      const utterance = new Float32Array(len);
      let off = 0;
      chunks.forEach((c) => {
        utterance.set(c, off);
        off += c.length;
      });
      const duration = speechMs;
      active = false;
      chunks = [];
      preRoll = [];
      silenceMs = 0;
      speechMs = 0;
      hotFrames = 0;
      if (duration >= minSpeechMs) {
        handleSpeechEnd(utterance);
      }
    }
  };
}

init();
