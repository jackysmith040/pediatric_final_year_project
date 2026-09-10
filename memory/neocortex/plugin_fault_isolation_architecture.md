---
neuron_id: plugin-fault-isolation-architecture
title: Plugin Fault Isolation Architecture
synaptic_weight: 45
corpus_callosum: hevc-poc-seek-dynamics
blindspot: false
summary: Decouples the invariant counting and tracking core from I/O exporters and visualizers via an exception-isolated PluginManager with bidirectional ControlSignals.
---

# Plugin Fault Isolation Architecture

## Core Principle
The demographic counting engine maintains an invariant mathematical state ($N = |K|$ distinct set cardinality). Any external reporting mechanism—whether writing CSV files, encoding MP4 streams, compiling JSON summaries, or rendering live OpenCV UI overlays—must be decoupled and isolated from this core.

## Mechanism
1. **PipelinePlugin Protocol**: Plugins implement lifecycle hooks (`on_start`, `on_frame`, `on_finish`, `on_error`, `close`) with priority ordering.
2. **Exception Barrier**: The `PluginManager` wraps all broadcasts in defensive try/except blocks. If a plugin throws a runtime exception, it is disabled and logged; the counting engine continues unhindered.
3. **Two-Way Control**: Plugins return non-blocking `ControlSignal` tokens (`CONTINUE`, `PAUSE`, `STEP`, `QUIT`, `SPEED_CHANGE`) allowing UI elements like keyboard shortcuts to steer playback without holding synchronous thread locks.
4. **Frame Copy Isolation**: Visualizer plugins receive an isolated copy of the image tensor (`annotated = frame.copy()`), preventing accidental in-place mutation of raw CCTV frames.
