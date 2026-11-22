ID: ISSUE-LOCAL-01M81052FFMC629SKEB0GDK2PN
Title: olala W2d-2: thread the V attention KV cache through the step host
Row: NEW-CANDIDATE-PORTS
State: OPEN
Kind: bug
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-11-22
Updated: 2026-11-22
Closed: -

## Problem

In W2d-1 the V-layer attention cache row at the current position stayed zero: the step host computed post-shift post-norm k/v into local arrays but never wrote the caller's cache, so later decode tokens could not attend earlier cross-token keys. The recorded limitation contradicted the cache contract the reference establishes (modeling_olala.py:619 raw shift cache, :643-673 post-norm attention cache).

## Resolution

-
