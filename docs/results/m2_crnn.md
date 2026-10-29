# M2 results: the CRNN against Tesseract

Run `main` (step 18000), 4,086,500 parameters, 260 classes; torch 2.15.0, macOS-27.2-arm64-arm-64bit-Mach-O, mps. CER in percent with a bootstrap 95% interval. Tesseract: best models, each set in its own language, page mode 7 (M1). The CRNN reads every set with one model and is never told the language.

## Headline

| Set | Images | Tesseract | CRNN, greedy | CRNN, beam | CRNN, beam + TinyGPT | CRNN against Tesseract | p |
|---|---:|---:|---:|---:|---:|---:|---:|
| synth_seen | 5736 | 4.2 (3.9 to 4.4) | 0.7 (0.7 to 0.8) | 0.7 (0.6 to 0.8) | 0.7 (0.6 to 0.8) | 3.5 fewer (3.7 fewer to 3.3 fewer) | < 0.001 |
| synth_unseen | 5728 | 3.2 (3.0 to 3.4) | 1.0 (0.9 to 1.0) | 0.9 (0.9 to 1.0) | 0.8 (0.8 to 0.9) | 2.4 fewer (2.5 fewer to 2.2 fewer) | < 0.001 |
| mozhi | 10173 | 6.5 (6.2 to 6.9) | 0.6 (0.5 to 0.7) | 0.6 (0.5 to 0.7) | 0.6 (0.5 to 0.7) | 5.9 fewer (6.3 fewer to 5.6 fewer) | < 0.001 |
| cord_word | 2356 | 19.9 (18.4 to 21.4) | 1.3 (1.1 to 1.5) | 1.3 (1.1 to 1.5) | 1.3 (1.1 to 1.6) | 18.5 fewer (20.0 fewer to 17.1 fewer) | < 0.001 |
| cord_line | 791 | 42.8 (39.5 to 46.1) | 2.8 (2.3 to 3.3) | 2.8 (2.3 to 3.3) | 2.7 (2.1 to 3.2) | 40.1 fewer (43.3 fewer to 36.9 fewer) | < 0.001 |

The comparison column is `beam_lm` against Tesseract on the same images: how many points of CER fewer or more the CRNN has, with the interval of that difference.

## All measures

| Set | Reader | CER | WER | Median CER | Catastrophic | Exact | Wrong script |
|---|---|---:|---:|---:|---:|---:|---:|
| synth_seen | tesseract | 4.2 | 10.3 | 0.0 | 0.6 | 59.3 | 7.9 |
| synth_seen | greedy | 0.7 | 3.0 | 0.0 | 0.0 | 88.3 | 0.1 |
| synth_seen | beam | 0.7 | 3.0 | 0.0 | 0.0 | 88.4 | 0.1 |
| synth_seen | beam_lm | 0.7 | 2.8 | 0.0 | 0.0 | 89.3 | 0.0 |
| synth_unseen | tesseract | 3.2 | 9.6 | 0.0 | 0.2 | 62.4 | 4.6 |
| synth_unseen | greedy | 1.0 | 4.7 | 0.0 | 0.0 | 82.3 | 0.0 |
| synth_unseen | beam | 0.9 | 4.6 | 0.0 | 0.0 | 82.5 | 0.0 |
| synth_unseen | beam_lm | 0.8 | 4.0 | 0.0 | 0.0 | 84.9 | 0.1 |
| mozhi | tesseract | 6.5 | 16.7 | 0.0 | 8.8 | 84.5 | 0.0 |
| mozhi | greedy | 0.6 | 1.9 | 0.0 | 0.1 | 98.2 | 0.0 |
| mozhi | beam | 0.6 | 1.9 | 0.0 | 0.1 | 98.2 | 0.0 |
| mozhi | beam_lm | 0.6 | 2.0 | 0.0 | 0.1 | 98.1 | 0.0 |
| cord_word | tesseract | 19.9 | 41.0 | 0.0 | 18.5 | 62.9 | 0.0 |
| cord_word | greedy | 1.3 | 5.8 | 0.0 | 0.5 | 94.9 | 0.0 |
| cord_word | beam | 1.3 | 6.0 | 0.0 | 0.5 | 94.8 | 0.0 |
| cord_word | beam_lm | 1.3 | 6.1 | 0.0 | 0.6 | 94.6 | 0.0 |
| cord_line | tesseract | 42.8 | 60.1 | 20.0 | 37.4 | 28.8 | 0.0 |
| cord_line | greedy | 2.8 | 11.9 | 0.0 | 0.1 | 78.8 | 0.0 |
| cord_line | beam | 2.8 | 11.8 | 0.0 | 0.1 | 79.1 | 0.0 |
| cord_line | beam_lm | 2.7 | 11.3 | 0.0 | 0.1 | 79.1 | 0.0 |

## Damage, by condition (synthetic lines, CER; beam_lm against Tesseract)

| Set | Reader | clean | blur | noise | low_dpi | jpeg | skew | shadow | photo |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| synth_seen | tesseract | 2.7 | 3.0 | 5.8 | 3.0 | 3.1 | 4.5 | 4.7 | 6.5 |
| synth_seen | beam_lm | 0.3 | 0.5 | 0.3 | 0.5 | 0.3 | 2.7 | 0.4 | 0.4 |
| synth_unseen | tesseract | 2.2 | 3.0 | 3.6 | 2.8 | 2.4 | 3.2 | 3.7 | 4.7 |
| synth_unseen | beam_lm | 0.5 | 0.6 | 0.5 | 0.5 | 0.5 | 3.0 | 0.5 | 0.6 |

## By script (synthetic lines, CER; beam_lm against Tesseract)

| Set | Reader | deva | latn | mixed |
|---|---|---:|---:|---:|
| synth_seen | tesseract | 4.4 | 1.9 | 6.5 |
| synth_seen | beam_lm | 0.6 | 0.3 | 1.5 |
| synth_unseen | tesseract | 2.8 | 1.7 | 6.5 |
| synth_unseen | beam_lm | 0.8 | 0.4 | 1.6 |

## TinyGPT rescoring (weights chosen on validation)

| Group | Validation images | lam | beta | Validation edits, plain beam | With TinyGPT |
|---|---:|---:|---:|---:|---:|
| deva | 3610 | 0.4 | 2.0 | 217 | 185 |
| latn | 2215 | 0.3 | 0.0 | 289 | 276 |

## Speed on this machine (images per second, greedy decoding)

| Set | GPU | CPU, all cores | Tesseract (M1) |
|---|---:|---:|---:|
| synth_seen | 228 | 58 | 32 |
| synth_unseen | 272 | 74 | 33 |
| mozhi | 900 | 288 | 511 |
| cord_word | 708 | 206 | 538 |
| cord_line | 212 | 55 | 215 |

## Characters the alphabet lacks (share of each test set's characters)

| Set | Missing | Most common |
|---|---:|---|
| synth_seen | 0.113% | ë (80) ĉ (24) ŭ (24) ĝ (24) ĥ (16) |
| synth_unseen | 0.105% | ë (80) ĉ (24) ŭ (24) ĝ (24) ĥ (16) |
| mozhi | 0.000% |  |
| cord_word | 0.000% |  |
| cord_line | 0.000% |  |

## What the CRNN gets wrong in Hindi (beam_lm: Mozhi plus clean synthetic Devanagari lines)

| Substituted (true > read) | Times | Dropped | Times | Invented | Times |
|---|---:|---|---:|---|---:|
| “ > " | 11 | ा | 11 | ा | 8 |
| ू > ु | 11 | ‍ | 6 | (space) | 5 |
| ु > ू | 8 | . | 5 | ़ | 4 |
| : > ः | 7 | (space) | 5 | ं | 4 |
| ि > े | 6 | ” | 5 | . | 4 |
| ' > " | 4 | ं | 4 | ः | 3 |
| ' > ’ | 4 | ' | 3 | र | 3 |
| – > - | 4 | ् | 3 | ् | 3 |
| ” > " | 4 | " | 3 | । | 2 |
| घ > ध | 3 | ‌ | 2 | ु | 2 |
| ल > न | 3 | ’ | 2 | ‌ | 1 |
| ख > ब | 3 | र | 1 | ओ | 1 |
| . > 0 | 3 | ‘ | 1 | ृ | 1 |
| व > द | 3 | क | 1 | ढ | 1 |
| ' > ‘ | 2 | ू | 1 | ' | 1 |
