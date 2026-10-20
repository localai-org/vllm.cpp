ID: ISSUE-LOCAL-01M4VPQYYNYHKHYBMA5S203KAH
Title: Build video token replacement sequence
Row: MODEL-MM-gemma4-mm-gemma4-for-conditional-generation
State: OPEN
Kind: feature
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-10-13
Updated: 2026-10-13
Closed: -

## Problem

The Python get_video_repl in gemma4_mm.py builds the interleaved token sequence for video frames: for each frame, format mm:ss timestamp, encode the prefix string, append boi, video_token*N, eoi. The C++ side has no equivalent. Video frames are decomposed into timestamped image frames processed at max_soft_tokens=70, with mm:ss timestamps between frames.

## Resolution

-
