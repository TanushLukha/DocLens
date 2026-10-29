# M2 results: the CRNN against Tesseract

Run `synthetic` (step 17000), 4,086,500 parameters, 260 classes; torch 2.15.0, macOS-27.2-arm64-arm-64bit-Mach-O, mps. CER in percent with a bootstrap 95% interval. Tesseract: best models, each set in its own language, page mode 7 (M1). The CRNN reads every set with one model and is never told the language.

## Headline

| Set | Images | Tesseract | CRNN, greedy | CRNN, beam | CRNN, beam + TinyGPT | CRNN against Tesseract | p |
|---|---:|---:|---:|---:|---:|---:|---:|
| synth_seen | 5736 | 4.2 (3.9 to 4.4) | 0.5 (0.5 to 0.6) | 0.5 (0.5 to 0.6) | 0.5 (0.4 to 0.5) | 3.7 fewer (3.9 fewer to 3.5 fewer) | < 0.001 |
| synth_unseen | 5728 | 3.2 (3.0 to 3.4) | 0.8 (0.7 to 0.8) | 0.7 (0.7 to 0.8) | 0.7 (0.6 to 0.7) | 2.5 fewer (2.7 fewer to 2.4 fewer) | < 0.001 |
| mozhi | 10173 | 6.5 (6.2 to 6.9) | 4.6 (4.3 to 4.9) | 4.6 (4.3 to 4.9) | 3.9 (3.6 to 4.1) | 2.7 fewer (3.0 fewer to 2.4 fewer) | < 0.001 |
| cord_word | 2356 | 19.9 (18.4 to 21.4) | 19.0 (17.8 to 20.2) | 19.0 (17.8 to 20.3) | 16.8 (15.6 to 18.0) | 3.1 fewer (4.4 fewer to 1.8 fewer) | < 0.001 |
| cord_line | 791 | 42.8 (39.5 to 46.1) | 21.1 (19.3 to 22.7) | 21.0 (19.2 to 22.7) | 19.9 (18.1 to 21.6) | 22.8 fewer (25.7 fewer to 20.1 fewer) | < 0.001 |

The comparison column is `beam_lm` against Tesseract on the same images: how many points of CER fewer or more the CRNN has, with the interval of that difference.

## All measures

| Set | Reader | CER | WER | Median CER | Catastrophic | Exact | Wrong script |
|---|---|---:|---:|---:|---:|---:|---:|
| synth_seen | tesseract | 4.2 | 10.3 | 0.0 | 0.6 | 59.3 | 7.9 |
| synth_seen | greedy | 0.5 | 2.3 | 0.0 | 0.0 | 91.0 | 0.1 |
| synth_seen | beam | 0.5 | 2.3 | 0.0 | 0.0 | 91.0 | 0.1 |
| synth_seen | beam_lm | 0.5 | 2.0 | 0.0 | 0.0 | 91.6 | 0.1 |
| synth_unseen | tesseract | 3.2 | 9.6 | 0.0 | 0.2 | 62.4 | 4.6 |
| synth_unseen | greedy | 0.8 | 3.8 | 0.0 | 0.0 | 84.9 | 0.1 |
| synth_unseen | beam | 0.7 | 3.8 | 0.0 | 0.0 | 85.1 | 0.1 |
| synth_unseen | beam_lm | 0.7 | 3.3 | 0.0 | 0.0 | 86.5 | 0.1 |
| mozhi | tesseract | 6.5 | 16.7 | 0.0 | 8.8 | 84.5 | 0.0 |
| mozhi | greedy | 4.6 | 14.3 | 0.0 | 6.5 | 86.4 | 0.4 |
| mozhi | beam | 4.6 | 14.4 | 0.0 | 6.5 | 86.4 | 0.4 |
| mozhi | beam_lm | 3.9 | 12.0 | 0.0 | 6.6 | 88.4 | 0.1 |
| cord_word | tesseract | 19.9 | 41.0 | 0.0 | 18.5 | 62.9 | 0.0 |
| cord_word | greedy | 19.0 | 51.2 | 0.0 | 14.1 | 56.8 | 0.2 |
| cord_word | beam | 19.0 | 51.7 | 0.0 | 14.2 | 56.8 | 0.2 |
| cord_word | beam_lm | 16.8 | 45.2 | 0.0 | 13.6 | 61.9 | 0.0 |
| cord_line | tesseract | 42.8 | 60.1 | 20.0 | 37.4 | 28.8 | 0.0 |
| cord_line | greedy | 21.1 | 53.3 | 12.5 | 13.0 | 28.3 | 1.3 |
| cord_line | beam | 21.0 | 53.1 | 12.0 | 13.0 | 28.7 | 1.3 |
| cord_line | beam_lm | 19.9 | 50.3 | 10.0 | 12.5 | 31.6 | 1.0 |

## Damage, by condition (synthetic lines, CER; beam_lm against Tesseract)

| Set | Reader | clean | blur | noise | low_dpi | jpeg | skew | shadow | photo |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| synth_seen | tesseract | 2.7 | 3.0 | 5.8 | 3.0 | 3.1 | 4.5 | 4.7 | 6.5 |
| synth_seen | beam_lm | 0.3 | 0.3 | 0.3 | 0.3 | 0.3 | 1.8 | 0.3 | 0.3 |
| synth_unseen | tesseract | 2.2 | 3.0 | 3.6 | 2.8 | 2.4 | 3.2 | 3.7 | 4.7 |
| synth_unseen | beam_lm | 0.4 | 0.6 | 0.5 | 0.5 | 0.4 | 2.1 | 0.4 | 0.5 |

## By script (synthetic lines, CER; beam_lm against Tesseract)

| Set | Reader | deva | latn | mixed |
|---|---|---:|---:|---:|
| synth_seen | tesseract | 4.4 | 1.9 | 6.5 |
| synth_seen | beam_lm | 0.4 | 0.2 | 1.2 |
| synth_unseen | tesseract | 2.8 | 1.7 | 6.5 |
| synth_unseen | beam_lm | 0.6 | 0.3 | 1.3 |

## TinyGPT rescoring (weights chosen on validation)

| Group | Validation images | lam | beta | Validation edits, plain beam | With TinyGPT |
|---|---:|---:|---:|---:|---:|
| deva | 3506 | 0.6 | 0.5 | 856 | 741 |
| latn | 2319 | 0.6 | 0.0 | 3309 | 3055 |

## Speed on this machine (images per second, greedy decoding)

| Set | GPU | CPU, all cores | Tesseract (M1) |
|---|---:|---:|---:|
| synth_seen | 225 | 59 | 32 |
| synth_unseen | 276 | 74 | 33 |
| mozhi | 904 | 282 | 511 |
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
| । > I | 346 | ा | 22 | ् | 63 |
| । > | | 217 | ़ | 14 | (space) | 33 |
| - > — | 66 | । | 11 | र | 20 |
| ि > े | 15 | ं | 11 | प | 12 |
| । > i | 14 | ् | 9 | ‘ | 11 |
| ू > ु | 14 | ‌ | 6 | ' | 11 |
| " > ‘ | 13 | ‍ | 6 | ं | 8 |
| ी > ो | 10 | (space) | 5 | । | 7 |
| ' > ‘ | 9 | . | 5 | ’ | 7 |
| ा > । | 9 | र | 4 | ृ | 5 |
| " > ’ | 9 | े | 4 | ा | 5 |
| “ > ‘ | 8 | ँ | 4 | ु | 4 |
| आ > अ | 8 | ’ | 4 | त | 4 |
| स > प | 8 | - | 3 | भ | 4 |
| र > ए | 8 | ू | 3 | - | 4 |
