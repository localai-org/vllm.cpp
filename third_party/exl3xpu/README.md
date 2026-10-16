EXL3 ESIMD kernels from 0xSero/exl3xpu, commit
`c59d9442aba8610188837e37724600f1517d7335`, MIT license (see LICENSE).

`exl3_esimd.h` is an unchanged copy of `csrc/exl3_esimd.h`, SHA-256
`aabdb13eddbcf7387dac2716b26e1005259a0a658d4b7d8b0e6b338499d0fdc6`.
Native VT launch/allocation wrappers live in `src/vt/xpu/xpu_exl3_smallm.cpp`;
they do not link Torch or use its tensor/stream wrappers.

`qwen3.8-27b-draft-vocab.json` is a separate unchanged metadata artifact from
`models/qwen3.8-27b-exl3-4.00bpw/draft_vocab.json` in 0xSero/exl3xpu at
`ce4c170cbf3d991bfc70c9c839ce236d7686cf54` (Git blob
`c541b0b3968134e4f96f37465bb3d80339957be3`). That revision's MIT license is
identical to LICENSE here. This 2654-byte file has SHA-256
`b4eadc088059190983fe0498af11864f5aaa2eaf2ec58ae9864f0715634d313d`.
It contains no model weights. Its metadata pin differs from the ESIMD source pin.
