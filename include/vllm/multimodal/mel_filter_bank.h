// THE SHARED MEL FILTERBANK — `transformers.audio_utils.mel_filter_bank` with
// `norm="slaney"` and `mel_scale="slaney"`, in double precision.
//
// WHY IT IS A SEAM AND NOT ANOTHER COPY. Before this header the tree carried a
// mel filterbank in `parakeet_audio_processor.cpp` (constructed) and consumed
// pre-dumped golden banks in the Whisper, Voxtral and Gemma-4 audio paths, with
// no shared audio front end anywhere. dots3-note's `dots` speech encoder
// (`nvidia/audio.py:96-107` @ `9035151d6`) needs the SAME construction at a
// different `(num_frequency_bins, num_mel_filters)`, and AGENTS.md's "Shared
// seams" says to extend the shared surface rather than write a parallel path.
// Parakeet's implementation was already parameterised rather than
// Parakeet-specific, so it is EXTRACTED here and both callers use it.
//
// Ported from transformers 5.3.0 `audio_utils.py`:
//   mel_filter_bank                 :453   (the whole construction)
//     the filter centres            :516-519
//     `np.linspace(0, sampling_rate // 2, num_frequency_bins)`  :528
//     the slaney area normalisation :532-535
//   hertz_to_mel(mel_scale="slaney")  :285-296
//   mel_to_hertz(mel_scale="slaney")  :321-331
//   _create_triangular_filter_bank   :356-375
//
// THE THREE CALLS IN THIS TREE, and they differ only in the first two arguments:
//   Parakeet  (257, 80, 0.0, 8000.0, 16000)  vllm/model_executor/models/
//                                            parakeet.py:159-167
//   dots3     (201, 128, 0.0, 8000.0, 16000) vllm/models/dots3_note/nvidia/
//                                            audio.py:98-106 @ 9035151d6
//   Whisper   (201, 128, 0.0, 8000.0, 16000) the same call, which is why the
//                                            fixture below can gate dots3
//
// THIS FUNCTION HAS A REAL ORACLE, WHICH ALMOST NOTHING ON THE dots3-note ROW
// DOES. `tests/vllm/multimodal/fixtures/voxtral_audio/voxtral_mel_filters_f32.bin`
// is a COMMITTED [201, 128] float32 matrix, dumped by
// `scripts/mm/a3_voxtral_oracle_capture.py:141-147` from
// `mistral_common.audio.mel_filter_bank(201, 128, 0.0, 8000.0, 16000)` — a third
// party's implementation of the same formula. `MelFilterBankSlaney(201, 128,
// 0.0, 8000.0, 16000)` reproduces all 25728 of its values BIT-FOR-BIT (max ULP
// difference 0), and `test_dots3_note_audio.cpp` asserts that. That single
// assertion settles HTK-versus-Slaney, the `norm` argument and the
// integer-divided-Nyquist detail at :528 outright, none of which a shape check
// or a loose tolerance could see.
//
// PRECISION. Every intermediate is `double` and the only rounding is the single
// `static_cast<float>` on the store, which is what makes the bit-exactness above
// reproducible rather than lucky. HF's own `ParakeetFeatureExtractor` calls
// `librosa.filters.mel` instead and says in a comment
// (feature_extraction_parakeet.py:83-93) that the ONLY difference is that
// `mel_filter_bank` works in float64 while librosa works in float32 — same
// formula, different rounding.
#pragma once

#include <vector>

namespace vllm::multimodal {

// HTK centres, triangles in Hz, norm=None; double [frequency, mel].
std::vector<double> MelFilterBankHtk(int frequency_bins, int mel_bins,
                                     double min_hz, double max_hz, int sample_rate);

// `audio_utils.hertz_to_mel(freq, mel_scale="slaney")` (:285-296) and its
// inverse `mel_to_hertz` (:321-331). Exposed because they are the piece a
// reviewer is most likely to want to check independently.
double HertzToMelSlaney(double hertz);
double MelToHertzSlaney(double mels);

// `mel_filter_bank(num_frequency_bins, num_mel_filters, min_frequency,
// max_frequency, sampling_rate, norm="slaney", mel_scale="slaney")`.
//
// Returned in UPSTREAM'S OWN ORIENTATION, `[num_frequency_bins,
// num_mel_filters]` row-major (`out[k * num_mel_filters + m]`) — the shape
// `mel_filter_bank` returns and the shape the Whisper/dots3 front end
// multiplies in (`WhisperAudioProcessor` indexes `mel_filters[k * n_mels + m]`,
// and `voxtral_mel_filters_f32.bin` is stored this way). A caller that wants the
// transpose asks for it by name below rather than transposing at the call site,
// so the two orientations cannot be confused by a reader.
//
// Throws when `num_frequency_bins < 2`, which is the one input the construction
// cannot express (`np.linspace` needs two points to have a step).
std::vector<float> MelFilterBankSlaney(int num_frequency_bins,
                                       int num_mel_filters,
                                       double min_frequency,
                                       double max_frequency, int sampling_rate);

// The same bank as `[num_mel_filters, num_frequency_bins]`
// (`out[m * num_frequency_bins + k]`) — the orientation both Parakeet upstreams
// multiply in (`torch.from_numpy(filter_bank.T)`, parakeet.py:168).
//
// A TRANSPOSE ROUNDS NOTHING. It reorders `float`s that were already rounded on
// the store above, so this is BYTE-IDENTICAL to what a caller computing directly
// into the transposed layout would get. That is what makes the Parakeet callers
// byte-identical by construction rather than by tolerance.
std::vector<float> MelFilterBankSlaneyTransposed(int num_frequency_bins,
                                                 int num_mel_filters,
                                                 double min_frequency,
                                                 double max_frequency,
                                                 int sampling_rate);

}  // namespace vllm::multimodal
