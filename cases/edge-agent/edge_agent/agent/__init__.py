"""Agent loop: the core reasoning engine.

Inspired by claude-code's query.ts / QueryEngine.ts separation:
  - query_loop: pure state machine, no I/O except through QueryDeps
  - ConversationEngine: owns message history and per-conversation state
"""
