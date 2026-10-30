# M3 results: whole pages

torch 2.15.0, macOS-27.2-arm64-arm-64bit-Mach-O, CRNN on mps, tesseract 5.5.3. Page CER in percent with a bootstrap 95% interval over pages; lines match at an IoU of 0.5; tau is Kendall's tau of the reading order of matched lines (1: the true order). Seconds per page for finding lines and for reading them (Tesseract's time is its whole run, on one core).

## test_seen (150 pages)

| Reader | Page CER | Against Tesseract | p | Precision | Recall | F1 | Splits | Merges | Tau | Seconds |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| tesseract | 12.2 (9.5 to 15.0) |  |  | 93.0 | 91.4 | 92.2 | 154 | 20 | 0.983 | 11.94 |
| tesseract+crnn | 10.6 (7.8 to 13.5) | 1.6 fewer (1.9 fewer to 1.3 fewer) | < 0.001 | 93.0 | 91.4 | 92.2 | 154 | 20 | 0.983 | 12.82 |
| classical+crnn | 4.8 (2.8 to 7.1) | 7.4 fewer (10.6 fewer to 3.9 fewer) | < 0.001 | 98.2 | 96.4 | 97.3 | 1 | 136 | 0.980 | 1.10 |
| doctr+crnn | 6.6 (4.3 to 9.3) | 5.6 fewer (9.2 fewer to 1.8 fewer) | 0.006 | 69.7 | 90.0 | 78.5 | 1470 | 25 | 0.974 | 1.18 |
| doctr rows+crnn | 44.9 (39.4 to 50.1) | 32.7 more (26.4 more to 38.7 more) | < 0.001 | 69.7 | 90.0 | 78.5 | 1470 | 25 | 0.766 | 1.18 |
| true+crnn | 0.1 (0.1 to 0.2) | 12.1 fewer (14.9 fewer to 9.4 fewer) | < 0.001 |  |  |  |  |  |  | 0.94 |

### test_seen: page CER by condition

| Reader | clean | noise | photo | shadow | skew |
|---|---:|---:|---:|---:|---:|
| tesseract | 2.8 | 3.8 | 18.5 | 24.1 | 11.8 |
| tesseract+crnn | 1.1 | 1.3 | 17.7 | 23.0 | 10.1 |
| classical+crnn | 3.2 | 3.4 | 9.8 | 3.3 | 4.3 |
| doctr+crnn | 1.3 | 1.3 | 10.3 | 1.4 | 18.8 |
| doctr rows+crnn | 39.5 | 39.5 | 51.9 | 39.5 | 54.3 |
| true+crnn | 0.1 | 0.1 | 0.1 | 0.1 | 0.1 |

### test_seen: page CER by script and columns

| Reader | columns 1 | columns 2 | deva | latn | mixed |
|---|---:|---:|---:|---:|---:|
| tesseract | 15.0 | 9.7 | 11.2 | 12.8 | 12.7 |
| tesseract+crnn | 13.4 | 8.2 | 8.9 | 12.1 | 11.1 |
| classical+crnn | 1.1 | 8.1 | 1.0 | 6.3 | 7.5 |
| doctr+crnn | 1.6 | 11.1 | 6.9 | 6.0 | 6.9 |
| doctr rows+crnn | 11.5 | 74.8 | 45.0 | 41.6 | 47.9 |
| true+crnn | 0.1 | 0.1 | 0.1 | 0.1 | 0.1 |

## test_unseen (150 pages)

| Reader | Page CER | Against Tesseract | p | Precision | Recall | F1 | Splits | Merges | Tau | Seconds |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| tesseract | 12.5 (9.9 to 15.4) |  |  | 93.9 | 92.0 | 92.9 | 86 | 20 | 0.971 | 12.15 |
| tesseract+crnn | 11.6 (8.8 to 14.4) | 1.0 fewer (1.1 fewer to 0.8 fewer) | < 0.001 | 93.9 | 92.0 | 92.9 | 86 | 20 | 0.971 | 13.04 |
| classical+crnn | 5.9 (3.7 to 8.2) | 6.7 fewer (10.0 fewer to 3.3 fewer) | < 0.001 | 98.9 | 97.7 | 98.3 | 1 | 82 | 0.974 | 1.11 |
| doctr+crnn | 7.6 (5.1 to 10.5) | 4.9 fewer (8.5 fewer to 1.2 fewer) | 0.011 | 67.5 | 89.3 | 76.9 | 1540 | 19 | 0.972 | 1.14 |
| doctr rows+crnn | 45.6 (40.2 to 50.6) | 33.0 more (27.3 more to 38.5 more) | < 0.001 | 67.5 | 89.3 | 76.9 | 1540 | 19 | 0.767 | 1.14 |
| true+crnn | 0.5 (0.4 to 0.6) | 12.0 fewer (14.9 fewer to 9.4 fewer) | < 0.001 |  |  |  |  |  |  | 0.92 |

### test_unseen: page CER by condition

| Reader | clean | noise | photo | shadow | skew |
|---|---:|---:|---:|---:|---:|
| tesseract | 4.9 | 6.7 | 20.6 | 23.9 | 6.7 |
| tesseract+crnn | 4.0 | 5.1 | 20.0 | 23.3 | 5.5 |
| classical+crnn | 4.8 | 5.2 | 10.5 | 3.6 | 5.2 |
| doctr+crnn | 2.9 | 2.7 | 10.0 | 2.6 | 20.0 |
| doctr rows+crnn | 40.1 | 40.0 | 51.0 | 39.9 | 56.9 |
| true+crnn | 0.5 | 0.5 | 0.5 | 0.5 | 0.5 |

### test_unseen: page CER by script and columns

| Reader | columns 1 | columns 2 | deva | latn | mixed |
|---|---:|---:|---:|---:|---:|
| tesseract | 12.7 | 12.5 | 14.3 | 12.4 | 10.8 |
| tesseract+crnn | 11.9 | 11.4 | 13.2 | 12.0 | 9.4 |
| classical+crnn | 1.8 | 9.4 | 2.2 | 7.1 | 8.6 |
| doctr+crnn | 1.9 | 12.6 | 9.2 | 7.2 | 6.3 |
| doctr rows+crnn | 12.7 | 74.3 | 46.4 | 42.9 | 47.4 |
| true+crnn | 0.5 | 0.5 | 0.7 | 0.1 | 0.7 |

## cord (100 pages)

| Reader | Page CER | Against Tesseract | p | Precision | Recall | F1 | Splits | Merges | Tau | Seconds |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| tesseract | 51.8 (44.6 to 59.5) |  |  | 38.9 | 39.3 | 39.1 | 127 | 18 | 1.000 | 0.47 |
| tesseract+crnn | 48.1 (40.6 to 56.3) | 3.6 fewer (5.8 fewer to 1.5 fewer) | < 0.001 | 38.9 | 39.3 | 39.1 | 127 | 18 | 1.000 | 0.79 |
| classical+crnn | 44.2 (34.6 to 55.6) | 7.6 fewer (17.6 fewer to 4.3 more) | 0.185 | 46.0 | 68.6 | 55.1 | 124 | 49 | 1.000 | 0.60 |
| doctr+crnn | 47.1 (42.0 to 51.7) | 4.6 fewer (13.4 fewer to 3.1 more) | 0.257 | 10.4 | 20.5 | 13.8 | 643 | 0 | 0.975 | 0.48 |
| doctr rows+crnn | 5.6 (3.6 to 8.0) | 46.1 fewer (53.6 fewer to 39.5 fewer) | < 0.001 | 10.4 | 20.5 | 13.8 | 643 | 0 | 1.000 | 0.48 |
| true+crnn | 1.9 (1.3 to 2.5) | 49.9 fewer (57.6 fewer to 42.9 fewer) | < 0.001 |  |  |  |  |  |  | 0.37 |

