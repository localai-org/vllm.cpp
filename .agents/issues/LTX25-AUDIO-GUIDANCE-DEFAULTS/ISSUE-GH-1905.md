ID: ISSUE-GH-1905
Title: **`verify_render.py` produced the FAIL verdict [#1510](https://github.com/mudler/vllm.cpp/issues/1510) rests on, and it is not in this repository.** Its only copy is on a CIFS share, `/mnt/nas_share/rc/ltx25-fullmodel/job/verify_render.py`, sha256 `57cf92846506be961e3c6ab3c9198de5608d0e85bfd1eb992eb91dda1c0fa563`. It is unversioned, untested and unreachable from any gate here; `grep -rn 'envelope_cv'` over this tree returns nothing. Two properties bear on the verdicts it produced. Its `envelope_cv` rests on 20 non-overlapping 50 ms frames over a 1.01 s clip (`hop = max(int(sr * 0.05), 1)` at 48 kHz over 48,480 frames) against a FAIL threshold of 0.10, with no confidence interval computed or reported. Its `active_fraction` threshold is RELATIVE, at `env.max() * 10 ** (-40 / 20.0)`, so a constant-level signal reads 1.0 at any absolute level, which is why both arms of the #1510 A/B read exactly 1.0 while the issue table shows a dash against 1.0. It also downmixes to mono before every headline metric, so its `peak_dbfs` of -0.287 is a mixdown value against a stereo peak sample of 32767. Until the instrument is in the tree with a test and a gate, no audio PASS or FAIL from this campaign is a gate result. Listed under `## Owed` in [`ltx25-audio-guidance-defaults.md`](../../specs/ltx25-audio-guidance-defaults.md)
Row: LTX25-AUDIO-GUIDANCE-DEFAULTS
State: UNKNOWN
Kind: bug
GitHub: 1905
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:726`

### Frozen archive evidence

> | [#1905](https://github.com/mudler/vllm.cpp/issues/1905) | `LTX25-AUDIO-GUIDANCE-DEFAULTS` | **`verify_render.py` produced the FAIL verdict [#1510](https://github.com/mudler/vllm.cpp/issues/1510) rests on, and it is not in this repository.** Its only copy is on a CIFS share, `/mnt/nas_share/rc/ltx25-fullmodel/job/verify_render.py`, sha256 `57cf92846506be961e3c6ab3c9198de5608d0e85bfd1eb992eb91dda1c0fa563`. It is unversioned, untested and unreachable from any gate here; `grep -rn 'envelope_cv'` over this tree returns nothing. Two properties bear on the verdicts it produced. Its `envelope_cv` rests on 20 non-overlapping 50 ms frames over a 1.01 s clip (`hop = max(int(sr * 0.05), 1)` at 48 kHz over 48,480 frames) against a FAIL threshold of 0.10, with no confidence interval computed or reported. Its `active_fraction` threshold is RELATIVE, at `env.max() * 10 ** (-40 / 20.0)`, so a constant-level signal reads 1.0 at any absolute level, which is why both arms of the #1510 A/B read exactly 1.0 while the issue table shows a dash against 1.0. It also downmixes to mono before every headline metric, so its `peak_dbfs` of -0.287 is a mixdown value against a stereo peak sample of 32767. Until the instrument is in the tree with a test and a gate, no audio PASS or FAIL from this campaign is a gate result. Listed under `## Owed` in [`ltx25-audio-guidance-defaults.md`](../../specs/ltx25-audio-guidance-defaults.md) | bug |

## Resolution

-
