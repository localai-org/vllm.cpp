# B70 optimization campaign measurements, 15 October 2026

Values are representative medians from each wave, not one paired causal chain.
Each individual wave retains its counterbalanced controls. Peak values are tracked
allocator bytes, measured only in the separate Wave1 P4096/O64 diagnostics.

| Variant | MTP0 prefill s | MTP0 decode tok/s | MTP0 total s | MTP3 prefill s | MTP3 decode tok/s | MTP3 total s | Peak tracked bytes (M0 / M3) |
|---|---:|---:|---:|---:|---:|---:|---|
| Current control (Wave0) | 1.985 | 25.72 | 11.900 | 2.062 | 58.27 | 6.438 | NOT RUN |
| Wave1A selected | 1.830 | 25.35 | 11.882 | 1.882 | 57.86 | 6.289 | 27,002,444,580 / 28,602,089,987 |
| Wave2 subgroup selected | NOT RUN | NOT RUN | NOT RUN | 1.868 | 63.00 | 5.915 | NOT RUN |
| Wave3 selected | 1.834 | 27.31 | 11.172 | 1.888 | 63.33 | 5.915 | NOT RUN |
| Wave4 selected (explicit1) | 1.779 | 27.91 | 10.917 | 1.830 | 63.88 | 5.820 | NOT RUN |
| Final default-build confirmation | 1.752 | 28.14 | 10.821 | 1.823 | 64.26 | 5.794 | NOT RUN |

Wave1 and Wave3/4 selection use three paired blocks per mode. Wave2 alone has
one MTP3 pilot pair; MTP0 was not independently remeasured at that intermediate
checkpoint. The final default-build row is a separate one-worker-per-mode
confirmation with fresh Python both modes. See each wave report for exactness,
route admission, numerical differences, rejected candidates and limitations.

The configured maximum context and all workload/concurrency shapes are not
qualified by these bounded scores. The subgroup long natural-generation quality
check remains deferred by the developer. The campaign ends with measured
remaining gaps; strict or near parity was not achieved.
