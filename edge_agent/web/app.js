/* EdgeClaw — WebSocket communication + UI logic */

const PROTO = location.protocol === 'https:' ? 'wss' : 'ws';
let ws, mediaStream, audioCtx, camVideo;
let sending = false;
let camVisible = true;
const FRAME_EVERY_N = 5;
let frameCounter = 0;

/* ── State management ──────────────────────────────────── */

const STATES = {
  idle:   { text: 'IDLE',      sub: 'Ready',            cls: 'orb-idle'   },
  listen: { text: 'LISTENING', sub: 'Hearing you...',    cls: 'orb-listen' },
  think:  { text: 'THINKING',  sub: 'Processing...',     cls: 'orb-think'  },
  speak:  { text: 'SPEAKING',  sub: 'Responding...',     cls: 'orb-speak'  },
};

function setState(s) {
  const state = STATES[s] || STATES.idle;
  const wrap = document.querySelector('.orb-wrap');
  wrap.className = 'orb-wrap ' + state.cls;
  document.getElementById('statusText').textContent = state.text;
  document.getElementById('statusSub').textContent = state.sub;
}

/* ── Transcript ────────────────────────────────────────── */

function addLine(cls, text) {
  if (!text || !text.trim()) return;
  const el = document.getElementById('transcript');
  const d = document.createElement('div');
  d.className = 't-line ' + cls;
  d.textContent = text;
  el.appendChild(d);
  el.scrollTop = el.scrollHeight;

  const lines = el.querySelectorAll('.t-line');
  if (lines.length > 50) {
    lines[0].remove();
  }
}

/* ── Hints ─────────────────────────────────────────────── */

function addHint(text, type) {
  const el = document.getElementById('hints');
  const h = document.createElement('div');
  h.className = 'hint ' + (type || '');
  h.textContent = text;
  el.appendChild(h);
  setTimeout(() => { if (h.parentNode) h.remove(); }, 5000);

  while (el.children.length > 3) {
    el.firstElementChild.remove();
  }
}

/* ── Audio encoding ────────────────────────────────────── */

function encodeWav(samples, sr) {
  const buf = new ArrayBuffer(44 + samples.length * 2);
  const v = new DataView(buf);
  const s = (o, str) => { for (let i = 0; i < str.length; i++) v.setUint8(o + i, str.charCodeAt(i)); };
  s(0, 'RIFF'); v.setUint32(4, 36 + samples.length * 2, true); s(8, 'WAVE');
  s(12, 'fmt '); v.setUint32(16, 16, true); v.setUint16(20, 1, true);
  v.setUint16(22, 1, true); v.setUint32(24, sr, true); v.setUint32(28, sr * 2, true);
  v.setUint16(32, 2, true); v.setUint16(34, 16, true);
  s(36, 'data'); v.setUint32(40, samples.length * 2, true);
  for (let i = 0; i < samples.length; i++) {
    let s16 = Math.max(-1, Math.min(1, samples[i])) * 0x7FFF;
    v.setInt16(44 + i * 2, s16, true);
  }
  return buf;
}

function sendChunk(pcm) {
  if (!ws || ws.readyState !== 1) return;
  const wav = encodeWav(pcm, 16000);
  const b64audio = btoa(String.fromCharCode(...new Uint8Array(wav)));

  let b64frame = '';
  frameCounter++;
  if (frameCounter % FRAME_EVERY_N === 0 && camVideo && camVideo.videoWidth > 0 && camVisible) {
    const c = document.createElement('canvas');
    c.width = 320; c.height = 240;
    c.getContext('2d').drawImage(camVideo, 0, 0, 320, 240);
    b64frame = c.toDataURL('image/jpeg', 0.6).split(',')[1];
  }
  ws.send(JSON.stringify({ type: 'audio_chunk', audio: b64audio, frame: b64frame }));
}

/* ── Audio playback ────────────────────────────────────── */

function playAudio(chunks) {
  if (!audioCtx) return;
  setState('speak');
  for (const c of chunks) {
    try {
      const obj = typeof c === 'string' ? JSON.parse(c) : c;
      const raw = atob(obj.pcm);
      const arr = new Float32Array(raw.length / 4);
      const dv = new DataView(new Uint8Array([...raw].map(ch => ch.charCodeAt(0))).buffer);
      for (let i = 0; i < arr.length; i++) arr[i] = dv.getFloat32(i * 4, true);
      const buf = audioCtx.createBuffer(1, arr.length, obj.sr || 16000);
      buf.getChannelData(0).set(arr);
      const src = audioCtx.createBufferSource();
      src.buffer = buf;
      src.connect(audioCtx.destination);
      src.onended = () => { if (sending) setState('listen'); };
      src.start();
    } catch (e) { /* skip bad chunks */ }
  }
}

/* ── WebSocket message handling ────────────────────────── */

function handleMsg(m) {
  if (m.type === 'prepared') {
    sending = true;
    setState('listen');
    document.getElementById('dotS1').classList.add('ok');
  } else if (m.type === 'result') {
    if (m.text && m.text.trim()) {
      addLine('t-bot', m.text);
    }
    setState(m.is_listen ? 'listen' : 'speak');
  } else if (m.type === 'audio') {
    playAudio(m.chunks);
  } else if (m.type === 'agent_status') {
    setState('think');
    document.getElementById('dotS2').classList.add('busy');
    document.getElementById('dotS2').classList.remove('ok');
    addHint('System 2 thinking...', '');
  } else if (m.type === 'agent_result') {
    setState(sending ? 'listen' : 'idle');
    document.getElementById('dotS2').classList.remove('busy');
    document.getElementById('dotS2').classList.add('ok');
    addLine('t-agent', m.text);
    if (m.tools_used && m.tools_used.length > 0) {
      m.tools_used.forEach(t => addHint(t, 'tool'));
    }
    if (m.text && 'speechSynthesis' in window) {
      const u = new SpeechSynthesisUtterance(m.text);
      u.lang = 'zh-CN';
      u.rate = 1.1;
      speechSynthesis.speak(u);
    }
  } else if (m.type === 'error') {
    addLine('t-sys', 'Error: ' + m.error);
  } else if (m.type === 'stopped') {
    setState('idle');
  } else if (m.type === 'reset_done') {
    addLine('t-sys', 'Context reset');
  }
}

/* ── Start / Stop ──────────────────────────────────────── */

async function startSession() {
  try {
    mediaStream = await navigator.mediaDevices.getUserMedia({ audio: true, video: true });
    camVideo = document.getElementById('cam');
    camVideo.srcObject = mediaStream;

    audioCtx = new AudioContext({ sampleRate: 16000 });
    const src = audioCtx.createMediaStreamSource(mediaStream);
    const proc = audioCtx.createScriptProcessor(4096, 1, 1);
    let buf = [];
    proc.onaudioprocess = e => {
      if (!sending) return;
      const d = e.inputBuffer.getChannelData(0);
      buf.push(new Float32Array(d));
      if (buf.length >= 4) {
        const total = buf.reduce((a, b) => a + b.length, 0);
        const merged = new Float32Array(total);
        let off = 0;
        for (const b2 of buf) { merged.set(b2, off); off += b2.length; }
        buf = [];
        sendChunk(merged);
      }
    };
    src.connect(proc);
    proc.connect(audioCtx.destination);

    ws = new WebSocket(`${PROTO}://${location.host}/ws/duplex`);
    ws.onopen = () => {
      addLine('t-sys', 'Connected');
      ws.send(JSON.stringify({ type: 'prepare' }));
    };
    ws.onmessage = e => handleMsg(JSON.parse(e.data));
    ws.onerror = () => addLine('t-sys', 'Connection error');
    ws.onclose = () => {
      addLine('t-sys', 'Disconnected');
      sending = false;
      setState('idle');
    };

    if ('webkitSpeechRecognition' in window || 'SpeechRecognition' in window) {
      const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
      const sr = new SR();
      sr.continuous = true;
      sr.interimResults = false;
      sr.lang = 'zh-CN';
      sr.onresult = e => {
        for (let i = e.resultIndex; i < e.results.length; i++) {
          if (e.results[i].isFinal) {
            const t = e.results[i][0].transcript.trim();
            if (t) {
              addLine('t-user', t);
              ws.send(JSON.stringify({ type: 'user_text', text: t }));
            }
          }
        }
      };
      sr.onerror = () => {};
      sr.onend = () => { if (sending) sr.start(); };
      sr.start();
      window._sr = sr;
    }
  } catch (e) {
    addLine('t-sys', 'Error: ' + e.message);
  }
}

function stopSession() {
  sending = false;
  if (ws && ws.readyState === 1) ws.send(JSON.stringify({ type: 'stop' }));
  if (window._sr) try { window._sr.stop(); } catch (e) {}
  if (mediaStream) mediaStream.getTracks().forEach(t => t.stop());
  setState('idle');
}

function toggleMic() {
  const btn = document.getElementById('micBtn');
  const micIcon = document.getElementById('micIcon');
  const stopIcon = document.getElementById('stopIcon');

  if (!btn.classList.contains('active')) {
    btn.classList.add('active');
    micIcon.classList.add('hidden');
    stopIcon.classList.remove('hidden');
    startSession();
  } else {
    btn.classList.remove('active');
    micIcon.classList.remove('hidden');
    stopIcon.classList.add('hidden');
    stopSession();
  }
}

function resetCtx() {
  if (ws && ws.readyState === 1) ws.send(JSON.stringify({ type: 'reset' }));
}

/* ── Camera toggle ─────────────────────────────────────── */

document.getElementById('camToggle').addEventListener('click', () => {
  camVisible = !camVisible;
  document.getElementById('camContainer').style.opacity = camVisible ? '1' : '0.3';
});

/* ── Status check on load ──────────────────────────────── */

fetch('/api/status')
  .then(r => r.json())
  .then(d => {
    if (d.system1) document.getElementById('dotS1').classList.add('ok');
    if (d.system2) document.getElementById('dotS2').classList.add('ok');
  })
  .catch(() => {});
