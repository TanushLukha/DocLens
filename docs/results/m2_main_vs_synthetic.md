# M2: `main` against `synthetic` on the same test images

Both runs train the same model with the same settings for the same number of steps; `main` draws synthetic lines, Mozhi training words and CORD training crops (50, 30 and 20% of samples), `synthetic` draws synthetic lines only. Each reads with the decoder below, its TinyGPT weights chosen on validation for that run. CER in percent; `main` against `synthetic` image by image, with a bootstrap 95% interval of the difference and a p value.

## Headline (beam_lm)

| Set | Images | synthetic | main | main against synthetic | p |
|---|---:|---:|---:|---:|---:|
| synth_seen | 5736 | 0.48 | 0.68 | 0.20 more (0.16 more to 0.24 more) | < 0.001 |
| synth_unseen | 5728 | 0.68 | 0.83 | 0.16 more (0.12 more to 0.20 more) | < 0.001 |
| mozhi | 10173 | 3.86 | 0.63 | 3.23 fewer (3.45 fewer to 3.01 fewer) | < 0.001 |
| cord_word | 2356 | 16.78 | 1.34 | 15.44 fewer (16.61 fewer to 14.32 fewer) | < 0.001 |
| cord_line | 791 | 19.94 | 2.65 | 17.29 fewer (18.79 fewer to 15.78 fewer) | < 0.001 |

## By script, synthetic lines (beam_lm)

| Set | Script | Images | synthetic | main | main against synthetic | p |
|---|---|---:|---:|---:|---:|---:|
| synth_seen | deva | 3200 | 0.40 | 0.61 | 0.21 more (0.16 more to 0.26 more) | < 0.001 |
| synth_seen | latn | 1600 | 0.17 | 0.25 | 0.08 more (0.05 more to 0.12 more) | < 0.001 |
| synth_seen | mixed | 936 | 1.15 | 1.48 | 0.33 more (0.22 more to 0.46 more) | < 0.001 |
| synth_unseen | deva | 3200 | 0.63 | 0.78 | 0.15 more (0.10 more to 0.21 more) | < 0.001 |
| synth_unseen | latn | 1600 | 0.32 | 0.39 | 0.07 more (0.03 more to 0.12 more) | < 0.001 |
| synth_unseen | mixed | 928 | 1.33 | 1.64 | 0.30 more (0.18 more to 0.42 more) | < 0.001 |

## Every decoder

| Set | Decoder | synthetic | main | main against synthetic | p |
|---|---|---:|---:|---:|---:|
| synth_seen | greedy | 0.53 | 0.74 | 0.21 more (0.17 more to 0.25 more) | < 0.001 |
| synth_seen | beam | 0.52 | 0.73 | 0.21 more (0.17 more to 0.25 more) | < 0.001 |
| synth_seen | beam_lm | 0.48 | 0.68 | 0.20 more (0.16 more to 0.24 more) | < 0.001 |
| synth_unseen | greedy | 0.76 | 0.96 | 0.20 more (0.15 more to 0.24 more) | < 0.001 |
| synth_unseen | beam | 0.75 | 0.94 | 0.20 more (0.15 more to 0.24 more) | < 0.001 |
| synth_unseen | beam_lm | 0.68 | 0.83 | 0.16 more (0.12 more to 0.20 more) | < 0.001 |
| mozhi | greedy | 4.61 | 0.63 | 3.98 fewer (4.22 fewer to 3.73 fewer) | < 0.001 |
| mozhi | beam | 4.61 | 0.61 | 4.00 fewer (4.25 fewer to 3.76 fewer) | < 0.001 |
| mozhi | beam_lm | 3.86 | 0.63 | 3.23 fewer (3.45 fewer to 3.01 fewer) | < 0.001 |
| cord_word | greedy | 18.99 | 1.28 | 17.70 fewer (18.87 fewer to 16.54 fewer) | < 0.001 |
| cord_word | beam | 19.00 | 1.31 | 17.68 fewer (18.87 fewer to 16.53 fewer) | < 0.001 |
| cord_word | beam_lm | 16.78 | 1.34 | 15.44 fewer (16.61 fewer to 14.32 fewer) | < 0.001 |
| cord_line | greedy | 21.05 | 2.79 | 18.26 fewer (19.74 fewer to 16.75 fewer) | < 0.001 |
| cord_line | beam | 21.00 | 2.75 | 18.24 fewer (19.72 fewer to 16.74 fewer) | < 0.001 |
| cord_line | beam_lm | 19.94 | 2.65 | 17.29 fewer (18.79 fewer to 15.78 fewer) | < 0.001 |
