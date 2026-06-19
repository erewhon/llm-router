import type { ReactNode } from "react";
import type { Phase, Speaker, Status } from "../use-voice";

const dotForPhase: Record<Phase, string> = {
  listening: "bg-moshi",
  thinking: "bg-amber-400",
  speaking: "bg-expert",
};

export function PhaseBar({
  status,
  phase,
  speaker,
  model,
  armed,
}: {
  status: Status;
  phase: Phase;
  speaker: Speaker;
  model: string | null;
  armed: boolean;
}) {
  let label: ReactNode;
  let dot = dotForPhase[phase];

  if (status !== "connected") {
    label =
      status === "connecting"
        ? "Connecting…"
        : status === "error"
          ? "Error"
          : status === "ended"
            ? "Disconnected"
            : "Idle";
    dot = "bg-slate-500";
  } else if (phase === "thinking") {
    label = (
      <span className="thinking-dots">
        Asking the expert<span>.</span>
        <span>.</span>
        <span>.</span>
      </span>
    );
  } else if (phase === "speaking") {
    label = (
      <>
        Expert speaking
        {model && <span className="ml-1 text-emerald-300/80">· {model}</span>}
      </>
    );
  } else {
    label = armed ? "Armed — ask your question" : "Listening · Moshi";
  }

  return (
    <div className="flex items-center gap-2 rounded-full border border-slate-700/70 bg-slate-900/60 px-4 py-1.5 text-sm backdrop-blur">
      <span
        className={`h-2.5 w-2.5 rounded-full ${dot} ${phase === "speaking" ? "animate-pulse" : ""}`}
      />
      <span className="font-medium text-slate-200">{label}</span>
      {status === "connected" && phase === "listening" && !armed && (
        <span className="ml-1 text-xs text-slate-500">({speaker})</span>
      )}
    </div>
  );
}
