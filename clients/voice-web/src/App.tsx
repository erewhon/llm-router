import { Controls } from "./components/Controls";
import { PhaseBar } from "./components/PhaseBar";
import { Transcript } from "./components/Transcript";
import { useVoice } from "./use-voice";

export function App() {
  const v = useVoice();
  const triggerHint = v.config?.triggers?.length
    ? `Say “${v.config.triggers[0]} …” or type a question`
    : undefined;

  return (
    <div className="mx-auto flex h-full max-w-2xl flex-col px-4 py-4">
      <header className="flex items-center justify-between gap-3 pb-3">
        <div>
          <h1 className="text-lg font-semibold tracking-tight text-slate-100">
            Voice Assistant
          </h1>
          <p className="text-xs text-slate-500">
            Moshi conversation · expert handoff to the LLM fleet
            {v.config?.llm && (
              <span className="text-slate-600"> · {v.config.llm}</span>
            )}
          </p>
        </div>
        <PhaseBar
          status={v.status}
          phase={v.phase}
          speaker={v.speaker}
          model={v.model}
          armed={v.armed}
        />
      </header>

      <main className="min-h-0 flex-1 overflow-y-auto rounded-2xl border border-slate-800/80 bg-slate-950/40 px-4">
        <Transcript bubbles={v.transcript} />
      </main>

      {v.error && (
        <p className="mt-2 rounded-lg border border-rose-500/40 bg-rose-500/10 px-3 py-1.5 text-xs text-rose-300">
          {v.error}
        </p>
      )}

      <footer className="pt-4">
        <Controls
          status={v.status}
          phase={v.phase}
          muted={v.muted}
          armed={v.armed}
          micLevel={v.micLevel}
          triggerHint={triggerHint}
          onConnect={v.connect}
          onDisconnect={v.disconnect}
          onAsk={v.ask}
          onStop={v.stop}
          onToggleMute={v.toggleMute}
        />
      </footer>
    </div>
  );
}
