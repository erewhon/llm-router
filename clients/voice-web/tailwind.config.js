/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        user: "#38bdf8", // sky-400 — the human
        moshi: "#a78bfa", // violet-400 — conversational Moshi
        expert: "#34d399", // emerald-400 — the LLM-fleet expert
      },
      fontFamily: {
        mono: ["ui-monospace", "SFMono-Regular", "Menlo", "monospace"],
      },
    },
  },
  plugins: [],
};
