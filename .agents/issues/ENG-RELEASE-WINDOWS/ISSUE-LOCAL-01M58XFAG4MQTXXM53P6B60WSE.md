ID: ISSUE-LOCAL-01M58XFAG4MQTXXM53P6B60WSE
Title: windows gate: test_openai_api_server.exe fails fast at startup (0xC0000409) once the compile fixes landed
Row: ENG-RELEASE-WINDOWS
State: OPEN
Kind: bug
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-10-19
Updated: 2026-10-19
Closed: -

## Problem

After the -WX compile repairs (#3479, merged), the windows-msvc lanes proceed past the build and now die running the tests: test_openai_api_server.exe exits -1073740791 (0xC0000409 STATUS_STACK_BUFFER_OVERRUN/fail-fast) before doctest enumerates any case ('could not read a test-case count ... nothing to enumerate', scripts/build-windows-release.ps1:39 rethrow). Startup-class crash: static init, TLS, or early main. Pre-existing — masked behind the compile errors on every earlier run; none of the open PRs (#3464/#3469/#3475/#3477) touch the server. Investigation deferred: the team's focus is P150/Kolibri-1; the four PRs merge with this lane known-red. First probes when picked up: run the binary alone under the localiser with a debugger or stderr capture; suspect the doctest main + httplib/socket static init under MSVC; check for abort()/__fastfail in static initializers and missing CRT/manifest DLLs.

## Resolution

-
