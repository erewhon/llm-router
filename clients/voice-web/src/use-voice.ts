import { useCallback, useRef, useState } from "react";
import { AudioEngine } from "./audio-engine";
import {
  decode,
  encode,
  isVoiceEvent,
  type VoiceCommand,
  type VoiceEvent,
  type VoiceSource,
} from "./protocol";

export type Status = "idle" | "connecting" | "connected" | "ended" | "error";
export type Phase = "listening" | "thinking" | "speaking";
export type Speaker = "moshi" | "expert";

export type Bubble = { id: number; source: VoiceSource; text: string };
export type VoiceConfig = {
  triggers: string[];
  llm: string;
  filler: string;
  handoff_enabled: boolean;
};

// Default Moshi generation params (mirrors the upstream client) — the proxy
// forwards this query string upstream to moshi-backend.
const GEN_PARAMS: Record<string, string> = {
  text_temperature: "0.7",
  text_topk: "25",
  audio_temperature: "0.8",
  audio_topk: "250",
  pad_mult: "0",
  repetition_penalty_context: "64",
  repetition_penalty: "1.0",
};

const buildWsUrl = (): string => {
  const proto = window.location.protocol === "https:" ? "wss" : "ws";
  const url = new URL(`${proto}://${window.location.host}/api/chat`);
  for (const [k, v] of Object.entries(GEN_PARAMS)) url.searchParams.set(k, v);
  url.searchParams.set("text_seed", String(Math.floor(Math.random() * 1e6)));
  url.searchParams.set("audio_seed", String(Math.floor(Math.random() * 1e6)));
  return url.toString();
};

const phaseFromEvent = (p?: VoiceEvent["phase"]): Phase | null => {
  switch (p) {
    case "trigger":
    case "thinking":
      return "thinking";
    case "speaking":
      return "speaking";
    case "moshi":
    case "cancelled":
      return "listening";
    default:
      return null;
  }
};

export function useVoice() {
  const [status, setStatus] = useState<Status>("idle");
  const [phase, setPhase] = useState<Phase>("listening");
  const [speaker, setSpeaker] = useState<Speaker>("moshi");
  const [model, setModel] = useState<string | null>(null);
  const [transcript, setTranscript] = useState<Bubble[]>([]);
  const [muted, setMuted] = useState(false);
  const [armed, setArmed] = useState(false);
  const [config, setConfig] = useState<VoiceConfig | null>(null);
  const [micLevel, setMicLevel] = useState(0);
  const [error, setError] = useState<string | null>(null);

  const wsRef = useRef<WebSocket | null>(null);
  const engineRef = useRef<AudioEngine | null>(null);
  const srcRef = useRef<VoiceSource>("moshi"); // current assistant text owner
  const idRef = useRef(0);
  const levelThrottle = useRef(0);

  const nextId = () => ++idRef.current;

  // append to the last bubble if same source still streaming, else open a new one
  const appendStream = useCallback((source: VoiceSource, text: string) => {
    setTranscript((prev) => {
      const last = prev[prev.length - 1];
      if (last && last.source === source) {
        const copy = prev.slice();
        copy[copy.length - 1] = { ...last, text: last.text + text };
        return copy;
      }
      return [...prev, { id: nextId(), source, text }];
    });
  }, []);

  const pushTurn = useCallback((source: VoiceSource, text: string) => {
    setTranscript((prev) => [...prev, { id: nextId(), source, text }]);
  }, []);

  const handleVoiceEvent = useCallback(
    (ev: VoiceEvent) => {
      // a discrete user turn (from the server's STT tap)
      if (ev.source === "user" && ev.text) {
        pushTurn("user", ev.text);
        return;
      }
      if (ev.model) setModel(ev.model);
      const nextPhase = phaseFromEvent(ev.phase);
      if (nextPhase) setPhase(nextPhase);
      if (
        ev.phase === "trigger" ||
        ev.phase === "thinking" ||
        ev.phase === "speaking"
      ) {
        srcRef.current = "expert";
        setSpeaker("expert");
        setArmed(false);
      } else if (ev.phase === "moshi" || ev.phase === "cancelled") {
        srcRef.current = "moshi";
        setSpeaker("moshi");
      }
    },
    [pushTurn],
  );

  const onMessage = useCallback(
    (raw: ArrayBuffer) => {
      const msg = decode(new Uint8Array(raw));
      switch (msg.type) {
        case "handshake":
          setStatus("connected");
          engineRef.current?.resetPlayback();
          break;
        case "audio":
          engineRef.current?.decode(msg.data);
          break;
        case "text":
          appendStream(srcRef.current, msg.data);
          break;
        case "metadata":
          if (isVoiceEvent(msg.data)) handleVoiceEvent(msg.data);
          break;
        case "error":
          setError(msg.data);
          break;
      }
    },
    [appendStream, handleVoiceEvent],
  );

  const send = useCallback((cmd: VoiceCommand) => {
    const ws = wsRef.current;
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(encode({ type: "metadata", data: cmd }));
    }
  }, []);

  const connect = useCallback(async () => {
    if (status === "connecting" || status === "connected") return;
    setError(null);
    setStatus("connecting");
    setTranscript([]);
    srcRef.current = "moshi";
    setSpeaker("moshi");
    setPhase("listening");
    try {
      const cfg = await fetch("/api/voice/config")
        .then((r) => (r.ok ? r.json() : null))
        .catch(() => null);
      if (cfg) setConfig(cfg);

      const engine = new AudioEngine({
        onMicChunk: (pages) => {
          const ws = wsRef.current;
          if (ws && ws.readyState === WebSocket.OPEN) {
            ws.send(encode({ type: "audio", data: pages }));
          }
        },
        onMicLevel: (rms) => {
          const now = performance.now();
          if (now - levelThrottle.current > 66) {
            levelThrottle.current = now;
            setMicLevel(rms);
          }
        },
      });
      engineRef.current = engine;
      await engine.start(); // mic permission + audio graph (must follow a user gesture)

      const ws = new WebSocket(buildWsUrl());
      ws.binaryType = "arraybuffer";
      ws.onmessage = (e) => onMessage(e.data as ArrayBuffer);
      ws.onclose = () => {
        setStatus((s) => (s === "error" ? s : "ended"));
        void engineRef.current?.stop();
        engineRef.current = null;
      };
      ws.onerror = () => setError("websocket error");
      wsRef.current = ws;
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
      setStatus("error");
      await engineRef.current?.stop();
      engineRef.current = null;
    }
  }, [status, onMessage]);

  const disconnect = useCallback(() => {
    wsRef.current?.close();
    wsRef.current = null;
  }, []);

  const ask = useCallback(
    (query?: string) => {
      const q = (query ?? "").trim();
      send(q ? { cmd: "ask", query: q } : { cmd: "ask" });
      if (!q) setArmed(true); // armed: the next spoken turn becomes the question
    },
    [send],
  );

  const stop = useCallback(() => send({ cmd: "stop" }), [send]);

  const toggleMute = useCallback(() => {
    setMuted((m) => {
      const next = !m;
      engineRef.current?.setMuted(next);
      return next;
    });
  }, []);

  return {
    status,
    phase,
    speaker,
    model,
    transcript,
    muted,
    armed,
    config,
    micLevel,
    error,
    connect,
    disconnect,
    ask,
    stop,
    toggleMute,
  };
}
