import { memo, useEffect, useRef } from "react";
import type { Bubble } from "../use-voice";

const META: Record<
  Bubble["source"],
  { label: string; cls: string; align: string }
> = {
  user: {
    label: "You",
    cls: "border-user/40 bg-user/10 text-sky-100",
    align: "items-end",
  },
  moshi: {
    label: "Moshi",
    cls: "border-moshi/40 bg-moshi/10 text-violet-100",
    align: "items-start",
  },
  expert: {
    label: "🧠 Expert",
    cls: "border-expert/40 bg-expert/10 text-emerald-100",
    align: "items-start",
  },
};

function TranscriptImpl({ bubbles }: { bubbles: Bubble[] }) {
  const endRef = useRef<HTMLDivElement>(null);
  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [bubbles]);

  if (bubbles.length === 0) {
    return (
      <div className="flex h-full items-center justify-center text-center text-sm text-slate-500">
        <p>
          Start talking — Moshi is listening.
          <br />
          Say “ask the expert …” to hand off to the LLM fleet.
        </p>
      </div>
    );
  }

  return (
    <div className="flex flex-col gap-3 py-2">
      {bubbles.map((b) => {
        const m = META[b.source];
        return (
          <div key={b.id} className={`flex flex-col ${m.align}`}>
            <span className="mb-0.5 px-1 text-[10px] uppercase tracking-wider text-slate-500">
              {m.label}
            </span>
            <div
              className={`max-w-[85%] whitespace-pre-wrap rounded-2xl border px-3 py-2 text-sm leading-relaxed ${m.cls}`}
            >
              {b.text || "…"}
            </div>
          </div>
        );
      })}
      <div ref={endRef} />
    </div>
  );
}

// memoized: high-frequency mic-level updates in the parent must not re-render this
export const Transcript = memo(TranscriptImpl);
