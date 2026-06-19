// Mic capture (opus-recorder -> opus pages) + jitter-buffered playback (opus
// decoder worker -> AudioWorklet). Distilled from the upstream Moshi client's
// useUserAudio/useServerAudio hooks into a framework-free class the React hook
// drives. The opus encode/decode + worklet buffer tuning are reused verbatim;
// only the wiring is ours.
import Recorder from "opus-recorder";
import workletUrl from "./audio-processor.ts?worker&url";

const DECODER_URL = "/assets/decoderWorker.min.js"; // copied from opus-recorder by copy-workers.mjs
const ENCODER_URL = "/assets/encoderWorker.min.js";

export type EngineCallbacks = {
  onMicChunk: (pages: Uint8Array) => void; // an opus page to ship as MT=1
  onMicLevel?: (rms: number) => void; // 0..1, for a level meter
};

export class AudioEngine {
  private ctx: AudioContext | null = null;
  private worklet: AudioWorkletNode | null = null;
  private decoder: Worker | null = null;
  private recorder: Recorder | null = null;
  private micSource: MediaStreamAudioSourceNode | null = null;
  private micStream: MediaStream | null = null;
  private analyser: AnalyserNode | null = null;
  private levelRAF = 0;
  private micDuration = 0;
  muted = false;

  constructor(private cb: EngineCallbacks) {}

  /** Must be called from a user gesture (mic permission + AudioContext resume). */
  async start(): Promise<void> {
    const ctx = new AudioContext();
    this.ctx = ctx;
    await ctx.resume();

    // playback worklet
    try {
      this.worklet = new AudioWorkletNode(ctx, "moshi-processor");
    } catch {
      await ctx.audioWorklet.addModule(workletUrl);
      this.worklet = new AudioWorkletNode(ctx, "moshi-processor");
    }
    this.worklet.connect(ctx.destination);

    // opus decoder worker -> worklet frames
    this.decoder = new Worker(DECODER_URL);
    this.decoder.onmessage = (e: MessageEvent) => {
      if (!e.data) return;
      const frame: Float32Array = e.data[0];
      this.worklet?.port.postMessage({
        frame,
        type: "audio",
        micDuration: this.micDuration,
      });
    };
    this.decoder.postMessage({
      command: "init",
      bufferLength: (960 * ctx.sampleRate) / 24000,
      decoderSampleRate: 24000,
      outputBufferSampleRate: ctx.sampleRate,
      resampleQuality: 0,
    });

    // mic -> opus pages
    await this.startMic(ctx);
  }

  private async startMic(ctx: AudioContext): Promise<void> {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    this.micStream = stream;
    this.micSource = ctx.createMediaStreamSource(stream);
    this.analyser = ctx.createAnalyser();
    this.analyser.fftSize = 512;
    this.micSource.connect(this.analyser);
    this.startLevelMeter();

    const recorder = new Recorder({
      mediaTrackConstraints: true,
      encoderPath: ENCODER_URL,
      bufferLength: Math.round((960 * ctx.sampleRate) / 24000),
      encoderFrameSize: 20,
      encoderSampleRate: 24000,
      maxFramesPerPage: 2,
      numberOfChannels: 1,
      recordingGain: 1,
      resampleQuality: 3,
      encoderComplexity: 0,
      encoderApplication: 2049,
      streamPages: true,
    });
    recorder.ondataavailable = (data: Uint8Array) => {
      this.micDuration = recorder.encodedSamplePosition / 48000;
      if (!this.muted) this.cb.onMicChunk(data);
    };
    this.recorder = recorder;
    await recorder.start();
  }

  private startLevelMeter(): void {
    if (!this.cb.onMicLevel || !this.analyser) return;
    const buf = new Uint8Array(this.analyser.frequencyBinCount);
    const tick = () => {
      if (!this.analyser) return;
      this.analyser.getByteTimeDomainData(buf);
      let sum = 0;
      for (let i = 0; i < buf.length; i++) {
        const v = (buf[i] - 128) / 128;
        sum += v * v;
      }
      this.cb.onMicLevel?.(Math.sqrt(sum / buf.length));
      this.levelRAF = requestAnimationFrame(tick);
    };
    this.levelRAF = requestAnimationFrame(tick);
  }

  /** Feed an inbound opus page (MT=1 payload) to the decoder. */
  decode(pages: Uint8Array): void {
    // copy: the worker transfers the buffer, and the ws frame may be a view
    const copy = pages.slice();
    this.decoder?.postMessage({ command: "decode", pages: copy }, [
      copy.buffer,
    ]);
  }

  /** Flush the playback jitter buffer (e.g. on (re)connect). */
  resetPlayback(): void {
    this.worklet?.port.postMessage({ type: "reset" });
  }

  setMuted(m: boolean): void {
    this.muted = m;
  }

  async stop(): Promise<void> {
    cancelAnimationFrame(this.levelRAF);
    try {
      await this.recorder?.stop();
    } catch {
      /* ignore */
    }
    this.micSource?.disconnect();
    this.micStream?.getTracks().forEach((t) => t.stop());
    this.decoder?.terminate();
    this.worklet?.disconnect();
    try {
      await this.ctx?.close();
    } catch {
      /* ignore */
    }
    this.ctx = null;
    this.worklet = null;
    this.decoder = null;
    this.recorder = null;
    this.micSource = null;
    this.micStream = null;
    this.analyser = null;
  }
}
