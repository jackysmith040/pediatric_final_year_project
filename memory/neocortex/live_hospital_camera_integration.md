---
neuron_id: live-hospital-camera-integration
title: Live Hospital Camera and RTSP Stream Ingestion
synaptic_weight: 42
corpus_callosum: orchestrator_ai
blindspot: false
summary: Architecture for polymorphic stream ingestion unifying pre-recorded CCTV MP4s, local USB webcams, and network RTSP streams under a shared generator protocol.
---

# Live Hospital Camera and RTSP Stream Ingestion

## Context
Deploying the pediatric counting engine in a live hospital environment requires real-time camera ingestion without altering downstream tracking or distinct counting algorithms.

## Design
1. The `LiveCameraReader` exposes identical generator semantics (`frames() -> (idx, frame)`) as `VideoReader`.
2. Hardware USB cameras use the DirectShow backend on Windows for near-zero startup latency.
3. Network RTSP streams connect via standard RTSP URIs with automatic reconnection buffering.
4. Timestamps synchronize with system wall-clock time rather than embedded CCTV filename strings.
