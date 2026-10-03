ID: ISSUE-LOCAL-01M3APZC6GKX9ME6AE6336D3VY
Title: Port fastino/GLiNER2.5-Decide SystemOne-class decision classifier
Row: MODEL-GLINER25-DECIDE
State: OPEN
Kind: feature
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-09-24
Updated: 2026-09-24
Closed: -

## Problem

GLiNER2.5-Decide (fastino/GLiNER2.5-Decide) is a 340M SystemOne-class decision classifier built on a DeBERTa-v3-large encoder with a classification head. It makes bounded-choice decisions in a single forward pass with no prompt template and no generated tokens. It supports single-label choice, multi-label classification, ordinal scoring, and yes/no (noul) decisions, and can score multiple heads simultaneously in one call. It reuses the /v1/systemone API from kev/laya. vLLM has no DeBERTa, no disentangled attention, and no GLiNER. The model shares the DeBERTa disentangled-attention dependency with MODEL-GLINER25 but uses a classification head instead of a NER pooler. Oracle: vllm-factory (ddickmann/vllm-factory 7d6ff68). CPU + GPU (CUDA).

## Resolution

-
