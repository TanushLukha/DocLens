# M1 results: Tesseract on Hindi and English

tesseract 5.5.3; 8 parallel processes; page mode 7. CER and WER in percent (lower is better); median CER per image; catastrophic: share of images with CER above 50%; exact: share read perfectly; wrong script: share of outputs with letters of a script the true text does not contain.

## Crop margin and page mode (Mozhi validation, 2000 words, best models, -l hin, CER)

Margin: border of paper colour on every side, as a fraction of the crop height.

| Margin | Page mode 7: CER | Catastrophic | Page mode 8: CER | Catastrophic |
|---:|---:|---:|---:|---:|
| 0.0 | 21.6 | 19.1 | 20.4 | 18.4 |
| 0.1 | 6.8 | 8.0 | 5.8 | 6.5 |
| 0.25 | 6.2 | 7.5 | 6.7 | 6.5 |
| 0.5 | 6.3 | 7.6 | 16.2 | 10.7 |
| 1.0 | 6.3 | 7.5 | 60.2 | 53.2 |

## Fast against best models (each set's own language, raw images)

| Set | Language | Models | Images | CER | WER | Median CER | Catastrophic | Exact | Wrong script | Images per second |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| synth_seen | hin+eng | fast | 5736 | 3.9 | 9.4 | 0.0 | 0.6 | 61.7 | 7.0 | 121 |
| synth_seen | hin+eng | best | 5736 | 4.2 | 10.3 | 0.0 | 0.6 | 59.3 | 7.9 | 32 |
| synth_unseen | hin+eng | fast | 5728 | 3.1 | 8.5 | 0.0 | 0.2 | 65.5 | 4.5 | 122 |
| synth_unseen | hin+eng | best | 5728 | 3.2 | 9.6 | 0.0 | 0.2 | 62.4 | 4.6 | 33 |
| mozhi | hin | fast | 10173 | 7.1 | 17.5 | 0.0 | 9.7 | 83.7 | 0.0 | 826 |
| mozhi | hin | best | 10173 | 6.5 | 16.7 | 0.0 | 8.8 | 84.5 | 0.0 | 511 |
| cord_word | eng | fast | 2356 | 21.6 | 47.9 | 0.0 | 19.9 | 58.5 | 0.0 | 1167 |
| cord_word | eng | best | 2356 | 19.9 | 41.0 | 0.0 | 18.5 | 62.9 | 0.0 | 538 |
| cord_line | eng | fast | 791 | 45.0 | 65.2 | 22.7 | 39.3 | 25.3 | 0.0 | 460 |
| cord_line | eng | best | 791 | 42.8 | 60.1 | 20.0 | 37.4 | 28.8 | 0.0 | 215 |

## Damage, by condition (synthetic lines, best models, hin+eng, raw images, CER)

| Set | Script | clean | blur | noise | low_dpi | jpeg | skew | shadow | photo |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| synth_seen | deva | 2.6 | 2.8 | 6.9 | 3.3 | 3.0 | 4.4 | 5.1 | 7.3 |
| synth_seen | latn | 1.3 | 1.3 | 1.7 | 1.1 | 1.4 | 2.2 | 2.2 | 3.8 |
| synth_seen | mixed | 5.1 | 5.6 | 7.7 | 5.0 | 5.6 | 8.1 | 7.1 | 8.1 |
| synth_unseen | deva | 1.6 | 2.5 | 3.7 | 2.3 | 2.0 | 2.6 | 3.5 | 4.5 |
| synth_unseen | latn | 1.2 | 1.5 | 1.3 | 1.5 | 1.1 | 2.1 | 1.6 | 2.9 |
| synth_unseen | mixed | 5.5 | 6.6 | 6.8 | 6.2 | 5.8 | 6.6 | 7.1 | 7.8 |

## Which language setting (best models, raw images; synthetic sets: clean lines only)

| Set | Script | CER, -l hin | CER, -l eng | CER, -l hin+eng | Wrong script, -l hin+eng |
|---|---|---:|---:|---:|---:|
| synth_seen | deva | 1.4 | 84.4 | 2.6 | 8.0 |
| synth_seen | latn | 86.4 | 0.5 | 1.3 | 1.5 |
| synth_seen | mixed | 25.2 | 59.1 | 5.1 | 0.0 |
| synth_unseen | deva | 0.8 | 83.5 | 1.6 | 3.8 |
| synth_unseen | latn | 87.3 | 0.7 | 1.2 | 2.0 |
| synth_unseen | mixed | 24.6 | 59.1 | 5.5 | 0.0 |
| mozhi | all | 6.5 | 100.5 | 9.7 | 4.1 |
| cord_word | all | 84.5 | 19.9 | 29.8 | 2.0 |
| cord_line | all | 82.2 | 42.8 | 49.4 | 11.4 |

## Cleaning the image first (best models, each set's own language, CER)

| Set | Condition | raw | scale | scale+denoise | scale+clahe | scale+otsu | scale+sauvola | scale+deskew | full |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| synth_seen | all | 4.2 | 5.0 | 4.3 | 4.4 | 7.5 | 6.2 | 4.8 | 5.4 |
| synth_seen | clean | 2.7 | 2.8 | 2.6 | 3.0 | 3.2 | 3.0 | 2.9 | 3.0 |
| synth_seen | blur | 3.0 | 3.1 | 3.3 | 3.2 | 5.7 | 6.4 | 3.1 | 8.5 |
| synth_seen | noise | 5.8 | 6.1 | 2.8 | 7.0 | 7.1 | 17.4 | 5.9 | 3.7 |
| synth_seen | low_dpi | 3.0 | 3.4 | 3.1 | 3.3 | 4.7 | 5.3 | 3.3 | 6.0 |
| synth_seen | jpeg | 3.1 | 3.3 | 3.0 | 4.6 | 3.8 | 3.5 | 3.2 | 3.5 |
| synth_seen | skew | 4.5 | 4.6 | 6.9 | 4.4 | 7.2 | 6.5 | 2.9 | 9.5 |
| synth_seen | shadow | 4.7 | 9.5 | 5.5 | 5.4 | 15.7 | 3.3 | 9.7 | 3.3 |
| synth_seen | photo | 6.5 | 6.9 | 6.9 | 4.5 | 12.6 | 4.2 | 7.1 | 5.5 |
| synth_unseen | all | 3.2 | 4.0 | 3.4 | 3.5 | 6.2 | 4.9 | 3.9 | 4.8 |
| synth_unseen | clean | 2.2 | 2.2 | 2.3 | 2.3 | 2.3 | 2.5 | 2.2 | 2.7 |
| synth_unseen | blur | 3.0 | 2.6 | 2.8 | 2.7 | 5.2 | 6.2 | 2.8 | 8.0 |
| synth_unseen | noise | 3.6 | 4.1 | 2.3 | 4.9 | 4.6 | 11.9 | 3.9 | 3.1 |
| synth_unseen | low_dpi | 2.8 | 2.6 | 3.0 | 2.7 | 3.8 | 4.3 | 2.6 | 5.8 |
| synth_unseen | jpeg | 2.4 | 2.5 | 2.4 | 3.6 | 2.8 | 2.6 | 2.4 | 2.7 |
| synth_unseen | skew | 3.2 | 4.1 | 5.3 | 3.7 | 5.4 | 5.4 | 2.7 | 8.3 |
| synth_unseen | shadow | 3.7 | 8.6 | 4.4 | 4.6 | 14.7 | 2.6 | 8.7 | 2.7 |
| synth_unseen | photo | 4.7 | 5.2 | 4.3 | 3.4 | 10.5 | 3.7 | 5.5 | 5.0 |
| mozhi | all | 6.5 | 6.3 | 6.5 | 7.6 | 6.7 | 7.0 | 6.1 | 7.8 |
| cord_word | all | 19.9 | 20.1 | 17.5 | 27.2 | 20.2 | 31.2 | 19.6 | 32.5 |
| cord_line | all | 42.8 | 43.0 | 40.2 | 45.8 | 42.6 | 47.4 | 41.3 | 46.3 |

## What Tesseract gets wrong in Hindi (best models, -l hin: Mozhi; hin+eng: clean synthetic Devanagari lines)

| Substituted (true > read) | Times | Dropped | Times | Invented | Times |
|---|---:|---|---:|---|---:|
| । > | | 320 | । | 211 |   | 103 |
| ै > े | 27 | ं | 43 | ् | 92 |
| ू > ु | 26 | र | 34 | - | 73 |
| स > प | 21 | ह | 34 | ं | 46 |
| ह > द | 19 | ् | 32 | | | 43 |
| ू > ् | 15 | े | 28 | र | 29 |
| ी > ो | 15 | ़ | 27 | द | 28 |
| " > “ | 14 | 1 | 27 | . | 17 |
| ि > े | 12 | ू | 24 | प | 16 |
| आ > ा | 11 | ा | 24 | ह | 14 |
| न > त | 11 | क | 21 | क | 13 |
| ह > ् | 11 | ै | 19 | त | 13 |
| , > | | 11 | . | 17 | े | 12 |
| ह > ड | 10 | म | 16 | ‌ | 12 |
| श > द | 8 | ी | 13 | ' | 11 |
