import { useState } from "react";
import type { Phase, Status } from "../use-voice";

export function Controls({
  status,
  phase,
  muted,
  armed,
  micLevel,
  triggerHint,
  onConnect,
  onDisconnect,
  onAsk,
  onStop,
  onToggleMute,
}: {
  status: Status;
  phase: Phase;
  muted: boolean;
  armed: boolean;
  micLevel: number;
  triggerHint?: string;
  onConnect: () => void;
  onDisconnect: () => void;
  onAsk: (q?: string) => void;
  onStop: () => void;
  onToggleMute: () => void;
}) {
  const [draft, setDraft] = useState("");
  const connected = status === "connected";
  const level = Math.min(100, Math.round(micLevel * 400));

  if (!connected) {
    return (
      <div className="flex flex-col items-center gap-3">
        <button
          onClick={onConnect}
          disabled={status === "connecting"}
          className="rounded-full bg-expert px-8 py-3 font-semibold text-slate-900 shadow-lg transition hover:brightness-110 disabled:opacity-50"
        >
          {status === "connecting"
            ? "Connecting…"
            : status === "ended"
              ? "Reconnect"
              : "Start conversation"}
        </button>
        <p className="text-xs text-slate-500">
          Mic access required · best in Chrome
        </p>
      </div>
    );
  }

  const submitAsk = () => {
    onAsk(draft);
    setDraft("");
  };

  return (
    <div className="flex w-full flex-col gap-3">
      {/* ask-the-expert row */}
      <div className="flex items-center gap-2">
        <input
          value={draft}
          onChange={(e) => setDraft(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && submitAsk()}
          placeholder={
            triggerHint ?? "Type a question, or press Ask and speak it"
          }
          className="min-w-0 flex-1 rounded-full border border-slate-700 bg-slate-900/70 px-4 py-2 text-sm outline-none placeholder:text-slate-600 focus:border-expert"
        />
        <button
          onClick={submitAsk}
          className={`whitespace-nowrap rounded-full px-4 py-2 text-sm font-medium transition ${
            armed
              ? "bg-amber-400 text-slate-900"
              : "bg-expert/90 text-slate-900 hover:brightness-110"
          }`}
          title="Send a question to the LLM-fleet expert (empty = arm the next spoken turn)"
        >
          {armed ? "Listening…" : "🧠 Ask the expert"}
        </button>
        <button
          onClick={onStop}
          disabled={phase === "listening"}
          className="rounded-full border border-rose-500/60 px-4 py-2 text-sm text-rose-300 transition enabled:hover:bg-rose-500/15 disabled:opacity-30"
          title="Interrupt the expert and return to Moshi"
        >
          Stop
        </button>
      </div>

      {/* mic meter + session controls */}
      <div className="flex items-center gap-3">
        <button
          onClick={onToggleMute}
          className={`rounded-full border px-3 py-1.5 text-sm transition ${
            muted
              ? "border-rose-500/60 text-rose-300"
              : "border-slate-700 text-slate-300 hover:bg-slate-800"
          }`}
        >
          {muted ? "🔇 Muted" : "🎙️ Mic on"}
        </button>
        <div className="h-2 flex-1 overflow-hidden rounded-full bg-slate-800">
          <div
            className={`h-full rounded-full transition-[width] duration-75 ${muted ? "bg-slate-600" : "bg-user"}`}
            style={{ width: `${muted ? 0 : level}%` }}
          />
        </div>
        <button
          onClick={onDisconnect}
          className="rounded-full border border-slate-700 px-4 py-1.5 text-sm text-slate-300 transition hover:bg-slate-800"
        >
          End
        </button>
      </div>
    </div>
  );
}
