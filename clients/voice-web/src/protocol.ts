// Moshi binary websocket framing (adapted from the upstream Kyutai client) plus
// our additive MT=4 voice.* status/command shapes (see server.py).
//
// Each frame = 1 type byte + payload:
//   0x00 handshake | 0x01 opus audio | 0x02 utf-8 text | 0x03 control
//   0x04 json metadata | 0x05 error | 0x06 ping | 0x07 colored text

export const MT = {
  HANDSHAKE: 0x00,
  AUDIO: 0x01,
  TEXT: 0x02,
  CONTROL: 0x03,
  META: 0x04,
  ERROR: 0x05,
  PING: 0x06,
  COLOREDTEXT: 0x07,
} as const;

export const CONTROL = {
  start: 0x00,
  endTurn: 0x01,
  pause: 0x02,
  restart: 0x03,
} as const;

export type WSMessage =
  | { type: "handshake" }
  | { type: "audio"; data: Uint8Array }
  | { type: "text"; data: string }
  | { type: "control"; action: keyof typeof CONTROL }
  | { type: "metadata"; data: unknown }
  | { type: "error"; data: string }
  | { type: "ping" }
  | { type: "unknown"; byte: number };

const enc = new TextEncoder();
const dec = new TextDecoder();

export const encode = (m: WSMessage): Uint8Array => {
  switch (m.type) {
    case "handshake":
      return new Uint8Array([MT.HANDSHAKE, 0x00, 0x00]);
    case "audio":
      return new Uint8Array([MT.AUDIO, ...m.data]);
    case "text":
      return new Uint8Array([MT.TEXT, ...enc.encode(m.data)]);
    case "control":
      return new Uint8Array([MT.CONTROL, CONTROL[m.action]]);
    case "metadata":
      return new Uint8Array([MT.META, ...enc.encode(JSON.stringify(m.data))]);
    case "error":
      return new Uint8Array([MT.ERROR, ...enc.encode(m.data)]);
    case "ping":
      return new Uint8Array([MT.PING]);
    default:
      return new Uint8Array([0xff]);
  }
};

export const decode = (data: Uint8Array): WSMessage => {
  const t = data[0];
  const payload = data.subarray(1);
  switch (t) {
    case MT.HANDSHAKE:
      return { type: "handshake" };
    case MT.AUDIO:
      return { type: "audio", data: payload };
    case MT.TEXT:
      return { type: "text", data: dec.decode(payload) };
    case MT.COLOREDTEXT:
      return { type: "text", data: dec.decode(payload.subarray(1)) };
    case MT.CONTROL: {
      const action = (Object.keys(CONTROL) as (keyof typeof CONTROL)[]).find(
        (k) => CONTROL[k] === payload[0],
      );
      return action
        ? { type: "control", action }
        : { type: "unknown", byte: t };
    }
    case MT.META:
      try {
        return { type: "metadata", data: JSON.parse(dec.decode(payload)) };
      } catch {
        return { type: "unknown", byte: t };
      }
    case MT.ERROR:
      return { type: "error", data: dec.decode(payload) };
    case MT.PING:
      return { type: "ping" };
    default:
      return { type: "unknown", byte: t };
  }
};

// ---- our MT=4 voice.* events (server -> browser) -------------------------------
export type VoicePhase =
  | "trigger"
  | "thinking"
  | "speaking"
  | "moshi"
  | "cancelled";
export type VoiceSource = "user" | "moshi" | "expert";

export type VoiceEvent = {
  kind: "voice";
  phase?: VoicePhase;
  source?: VoiceSource;
  model?: string;
  query?: string;
  text?: string;
};

export const isVoiceEvent = (d: unknown): d is VoiceEvent =>
  typeof d === "object" &&
  d !== null &&
  (d as { kind?: unknown }).kind === "voice";

// ---- our MT=4 commands (browser -> server) -------------------------------------
export type VoiceCommand = { cmd: "ask"; query?: string } | { cmd: "stop" };
