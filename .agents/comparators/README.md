# Implementation and performance comparators

This registry pins external implementations used for mechanism study or
performance comparison. A comparator is **not a correctness oracle** and cannot
supply expected output. The primary and secondary correctness-oracle policy in
[`AGENTS.md`](../../AGENTS.md) is unchanged.

Each `<id>.md` contains exactly one `comparator-pin` block. The checker requires
an implementation pin, a deployment-recipe pin, a one-line scope, and the fixed
labels `role = implementation-performance` and
`correctness = not-an-oracle`.

Run `python3 scripts/check-comparator-pins.py` to validate the registry.
