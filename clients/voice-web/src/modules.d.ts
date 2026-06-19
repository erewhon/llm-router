// opus-recorder ships no TS types. We only use the main-thread Recorder class with
// an `encoderPath` pointing at the worker we copy into /assets.
declare module "opus-recorder" {
  interface RecorderOptions {
    encoderPath?: string;
    mediaTrackConstraints?: boolean | MediaTrackConstraints;
    bufferLength?: number;
    encoderFrameSize?: number;
    encoderSampleRate?: number;
    maxFramesPerPage?: number;
    numberOfChannels?: number;
    recordingGain?: number;
    resampleQuality?: number;
    encoderComplexity?: number;
    encoderApplication?: number;
    streamPages?: boolean;
  }
  export default class Recorder {
    constructor(options?: RecorderOptions);
    encodedSamplePosition: number;
    ondataavailable: (data: Uint8Array) => void;
    onstart: () => void;
    onstop: () => void;
    sourceNode: MediaStreamAudioSourceNode;
    start(): Promise<void> | void;
    stop(): Promise<void> | void;
  }
}

// Vite worklet import: `?worker&url` yields the bundled worklet's URL string.
declare module "*?worker&url" {
  const url: string;
  export default url;
}
