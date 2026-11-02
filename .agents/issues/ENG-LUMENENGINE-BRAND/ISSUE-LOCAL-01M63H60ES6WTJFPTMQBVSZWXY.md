ID: ISSUE-LOCAL-01M63H60ES6WTJFPTMQBVSZWXY
Title: Rebrand to LumenEngine without breaking public interfaces
Row: ENG-LUMENENGINE-BRAND
State: OPEN
Kind: feature
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-10-29
Updated: 2026-10-29
Closed: -

## Problem

The product still uses vllm.cpp branding. Add LumenEngine names while preserving existing API, ABI, commands, configuration, build options, and installed consumers.

## Resolution

-

## Evidence

29 October 2026: additive header, shared CMake names, command aliases, and maintained product branding are implemented. [CPU packaging evidence](../../evidence/lumenengine-brand/SUMMARY.md) records real build-tree and relocated installed consumers, ABI smoke tests, and negative mutations. Fresh review and operator verification remain pending. Keep this issue open until the work lands.
