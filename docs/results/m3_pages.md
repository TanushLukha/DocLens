# M3 results: whole pages

torch 2.15.0, macOS-27.2-arm64-arm-64bit-Mach-O, CRNN on mps, tesseract 5.5.3. Page CER in percent with a bootstrap 95% interval over page designs (a design's five damaged pages drawn together; on CORD, receipts), and the comparison with Tesseract paired the same way; lines match at an IoU of 0.5; tau is Kendall's tau of the reading order of matched lines (1: the true order). Seconds per page for finding lines and for reading them (Tesseract's time is its whole run, on one core).

## test_seen (150 pages, 30 designs)

| Reader | Page CER | Against Tesseract | p | Precision | Recall | F1 | Splits | Merges | Tau | Seconds |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| tesseract | 11.4 (8.6 to 14.2) |  |  | 93.5 | 92.2 | 92.8 | 152 | 9 | 0.981 | 12.04 |
| tesseract+crnn | 10.0 (7.1 to 12.7) | 1.4 fewer (1.7 fewer to 1.2 fewer) | < 0.001 | 93.5 | 92.2 | 92.8 | 152 | 9 | 0.981 | 12.99 |
| classical+crnn | 0.4 (0.2 to 0.9) | 11.0 fewer (13.8 fewer to 8.1 fewer) | < 0.001 | 99.8 | 99.8 | 99.8 | 0 | 38 | 1.000 | 1.13 |
| doctr+crnn | 10.9 (5.1 to 17.4) | 0.5 fewer (7.2 fewer to 6.8 more) | 0.852 | 67.5 | 89.0 | 76.8 | 1612 | 17 | 0.945 | 1.21 |
| doctr rows+crnn | 44.5 (32.1 to 55.9) | 33.1 more (20.4 more to 45.0 more) | < 0.001 | 67.5 | 89.0 | 76.8 | 1612 | 17 | 0.766 | 1.21 |
| true+crnn | 0.1 (0.1 to 0.2) | 11.3 fewer (14.0 fewer to 8.5 fewer) | < 0.001 |  |  |  |  |  |  | 1.11 |

### test_seen: page CER by condition

| Reader | clean | noise | photo | shadow | skew |
|---|---:|---:|---:|---:|---:|
| tesseract | 2.8 | 4.0 | 22.9 | 18.6 | 8.7 |
| tesseract+crnn | 1.5 | 1.9 | 21.8 | 17.6 | 7.0 |
| classical+crnn | 0.4 | 0.4 | 0.5 | 0.4 | 0.4 |
| doctr+crnn | 6.6 | 7.4 | 12.5 | 7.3 | 20.8 |
| doctr rows+crnn | 39.1 | 39.1 | 50.0 | 39.1 | 55.3 |
| true+crnn | 0.1 | 0.1 | 0.1 | 0.1 | 0.1 |

### test_seen: page CER by script and columns

| Reader | columns 1 | columns 2 | deva | latn | mixed |
|---|---:|---:|---:|---:|---:|
| tesseract | 12.8 | 10.2 | 10.6 | 9.5 | 13.7 |
| tesseract+crnn | 11.3 | 8.7 | 8.8 | 8.6 | 12.2 |
| classical+crnn | 0.2 | 0.6 | 0.7 | 0.3 | 0.2 |
| doctr+crnn | 0.8 | 20.1 | 6.5 | 12.2 | 14.6 |
| doctr rows+crnn | 10.9 | 74.8 | 44.0 | 43.0 | 46.2 |
| true+crnn | 0.1 | 0.1 | 0.1 | 0.1 | 0.1 |

## test_unseen (150 pages, 30 designs)

| Reader | Page CER | Against Tesseract | p | Precision | Recall | F1 | Splits | Merges | Tau | Seconds |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| tesseract | 9.7 (7.2 to 12.1) |  |  | 93.2 | 93.6 | 93.4 | 177 | 10 | 0.986 | 11.40 |
| tesseract+crnn | 8.6 (6.0 to 11.0) | 1.1 fewer (1.2 fewer to 1.0 fewer) | < 0.001 | 93.2 | 93.6 | 93.4 | 177 | 10 | 0.986 | 12.30 |
| classical+crnn | 1.6 (0.6 to 2.8) | 8.1 fewer (11.2 fewer to 5.0 fewer) | < 0.001 | 99.3 | 98.3 | 98.8 | 0 | 52 | 0.998 | 1.13 |
| doctr+crnn | 12.0 (6.3 to 17.7) | 2.3 more (4.4 fewer to 8.7 more) | 0.521 | 66.2 | 88.6 | 75.8 | 1728 | 18 | 0.940 | 1.13 |
| doctr rows+crnn | 45.3 (33.0 to 56.4) | 35.6 more (22.2 more to 48.2 more) | < 0.001 | 66.2 | 88.6 | 75.8 | 1728 | 18 | 0.767 | 1.13 |
| true+crnn | 0.4 (0.2 to 0.6) | 9.3 fewer (11.8 fewer to 6.7 fewer) | < 0.001 |  |  |  |  |  |  | 0.91 |

### test_unseen: page CER by condition

| Reader | clean | noise | photo | shadow | skew |
|---|---:|---:|---:|---:|---:|
| tesseract | 1.6 | 2.7 | 19.2 | 15.5 | 9.3 |
| tesseract+crnn | 0.5 | 1.1 | 18.4 | 14.6 | 8.1 |
| classical+crnn | 0.7 | 1.0 | 4.7 | 1.0 | 0.5 |
| doctr+crnn | 6.1 | 5.9 | 16.8 | 4.9 | 26.1 |
| doctr rows+crnn | 39.8 | 39.8 | 50.8 | 39.8 | 56.2 |
| true+crnn | 0.4 | 0.4 | 0.4 | 0.4 | 0.4 |

### test_unseen: page CER by script and columns

| Reader | columns 1 | columns 2 | deva | latn | mixed |
|---|---:|---:|---:|---:|---:|
| tesseract | 12.8 | 6.8 | 9.0 | 8.2 | 11.6 |
| tesseract+crnn | 11.7 | 5.7 | 7.7 | 7.3 | 10.5 |
| classical+crnn | 0.5 | 2.5 | 1.7 | 0.9 | 2.0 |
| doctr+crnn | 1.2 | 21.5 | 11.0 | 10.3 | 14.4 |
| doctr rows+crnn | 11.6 | 75.2 | 45.6 | 43.2 | 46.7 |
| true+crnn | 0.4 | 0.4 | 0.8 | 0.1 | 0.2 |

## cord (100 pages, 100 designs)

| Reader | Page CER | Against Tesseract | p | Precision | Recall | F1 | Splits | Merges | Tau | Seconds |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| tesseract | 51.8 (44.6 to 59.5) |  |  | 38.9 | 39.3 | 39.1 | 127 | 18 | 1.000 | 0.45 |
| tesseract+crnn | 48.1 (40.6 to 56.3) | 3.6 fewer (5.8 fewer to 1.5 fewer) | < 0.001 | 38.9 | 39.3 | 39.1 | 127 | 18 | 1.000 | 0.78 |
| classical+crnn | 44.2 (34.6 to 55.6) | 7.6 fewer (17.6 fewer to 4.3 more) | 0.187 | 46.0 | 68.6 | 55.1 | 124 | 49 | 1.000 | 0.59 |
| doctr+crnn | 47.1 (42.0 to 51.7) | 4.6 fewer (13.4 fewer to 3.1 more) | 0.257 | 10.4 | 20.5 | 13.8 | 643 | 0 | 0.975 | 0.48 |
| doctr rows+crnn | 5.6 (3.6 to 8.0) | 46.1 fewer (53.6 fewer to 39.5 fewer) | < 0.001 | 10.4 | 20.5 | 13.8 | 643 | 0 | 1.000 | 0.48 |
| true+crnn | 1.9 (1.3 to 2.5) | 49.9 fewer (57.6 fewer to 42.9 fewer) | < 0.001 |  |  |  |  |  |  | 0.37 |

