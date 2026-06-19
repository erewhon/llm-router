"""Local voice-assistant pipeline: Moshi conversation + LLM handoff + TTS.

See DESIGN.md. Slice 1 (`handoff_tts`) implements the reasoning+speech half:
text -> router auto-router -> streamed LLM answer -> Orpheus TTS -> audio.
"""
