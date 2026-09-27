ID: ISSUE-GH-857
Title: The recorded llama.cpp pin `237ad9b96` names an object no remote carries: `git branch -r --contains 237ad9b96` is empty and it lives only on local branch `localai-paged` in the developer's checkout. `git describe --tags` returns `b9827-65-g237ad9b96`, so it is 65 of OUR OWN performance commits past upstream tag `b9827`, and the recorded `pin_label = b9892` came from `git rev-list --count`, which returns 9892. That label is not merely derived, it COLLIDES: stock reached tag `b9892` exactly 65 commits after the same base `b9827`, so upstream `b9892` also counts 9892 and resolves to `ee445f93d`, `git merge-base 237ad9b96 b9892` is `b9827`, and neither is an ancestor of the other. A reader checking out `b9892` to reproduce a number silently gets stock. That already happened here: `rpi5-a76-llamacpp-20260806.md` substituted stock `b9892` after finding the pin unobtainable while `cpu-x86-llamacpp-20260811.md` built the fork under the same label. The working tree at the pin also carried 27 uncommitted entries at +2279/-762, so the binaries came from a tree in no repository. `ORACLE-LLAMACPP-REPIN-STOCK` moves the record to stock `b10451` (`10bf611e5`) and drops `gateable` to `no` naming THIS issue, which still owes the build-and-run on dgx.casa that would make it `yes`. Spec [`oracle-llamacpp-repin-stock.md`](../../specs/oracle-llamacpp-repin-stock.md)
Row: ORACLE-LLAMACPP-REPIN-STOCK
State: UNKNOWN
Kind: bug
GitHub: 857
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:281`

### Frozen archive evidence

> | [#857](https://github.com/mudler/vllm.cpp/issues/857) | `ORACLE-LLAMACPP-REPIN-STOCK` | The recorded llama.cpp pin `237ad9b96` names an object no remote carries: `git branch -r --contains 237ad9b96` is empty and it lives only on local branch `localai-paged` in the developer's checkout. `git describe --tags` returns `b9827-65-g237ad9b96`, so it is 65 of OUR OWN performance commits past upstream tag `b9827`, and the recorded `pin_label = b9892` came from `git rev-list --count`, which returns 9892. That label is not merely derived, it COLLIDES: stock reached tag `b9892` exactly 65 commits after the same base `b9827`, so upstream `b9892` also counts 9892 and resolves to `ee445f93d`, `git merge-base 237ad9b96 b9892` is `b9827`, and neither is an ancestor of the other. A reader checking out `b9892` to reproduce a number silently gets stock. That already happened here: `rpi5-a76-llamacpp-20260806.md` substituted stock `b9892` after finding the pin unobtainable while `cpu-x86-llamacpp-20260811.md` built the fork under the same label. The working tree at the pin also carried 27 uncommitted entries at +2279/-762, so the binaries came from a tree in no repository. `ORACLE-LLAMACPP-REPIN-STOCK` moves the record to stock `b10451` (`10bf611e5`) and drops `gateable` to `no` naming THIS issue, which still owes the build-and-run on dgx.casa that would make it `yes`. Spec [`oracle-llamacpp-repin-stock.md`](../../specs/oracle-llamacpp-repin-stock.md) | bug |

## Resolution

-
