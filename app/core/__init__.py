"""Adaptive Operator Runtime Substrate — the execution core.

This module implements the agent execution loop that any sufficiently capable
model can operate through. The model is the brain; this is the body.

Core components:
    AgentLoop      — think → act → observe → verify cycle
    ToolExecutor   — tool execution with safety gates and risk evaluation
    Trajectory     — training-quality telemetry for every interaction
    SafetyGate     — risk classification and approval enforcement
    ExecutionState — mutable state tracking during a loop run
    SystemPrompt   — dynamic system prompt assembly from context + tools

Design principles:
    1. Model is the brain, runtime is the body — no hardcoded intelligence
    2. Everything observable — every step is training-quality telemetry
    3. Capabilities, not agents — the model orchestrates, tools execute
    4. Model-replaceable — build for v3/v4, not current Qwen3.5 limitations
"""
