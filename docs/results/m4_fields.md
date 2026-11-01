# M4 results: fields on receipts

torch 2.15.0, macOS-27.2-arm64-arm-64bit-Mach-O, mps. Field F1 in percent on CORD's 100 test receipts: a field counts when its category and its text (compared without spaces) are both right. Bootstrap 95% intervals over receipts; runs that differ only in their seed are pooled per receipt.

## Ceilings

| Input | Field F1 |
|---|---:|
| true words, perfect tags | 99.7 (99.0 to 100.0) |
| OCR words, tags of the true words under them | 85.2 (81.3 to 88.4) |

## Models

| Setting | Input | Runs | Field F1 | Each run | Validation F1 |
|---|---|---:|---:|---|---|
| layoutlmv3 (segment boxes, trained on true words) | true words | 1 | 94.5 (92.5 to 96.4) | 94.5 | 96.7 |
| layoutlmv3 (word boxes, trained on true words) | true words | 3 | 93.7 (91.4 to 95.9) | 93.6, 94.4, 93.1 | 95.6, 95.7, 95.6 |
| layoutlmv3 (word boxes, trained on true words) | OCR words | 3 | 80.2 (76.0 to 84.1) | 80.1, 80.4, 80.2 | 95.6, 95.7, 95.6 |
| layoutlmv3 (word boxes, trained on true words) | true words, one box for all | 3 | 58.2 (54.6 to 61.8) | 52.3, 62.2, 60.1 | 95.6, 95.7, 95.6 |
| layoutlmv3 (word boxes, trained on true words) | true words, blank image | 3 | 93.9 (91.5 to 96.0) | 93.9, 94.7, 93.0 | 95.6, 95.7, 95.6 |
| lilt (segment boxes, trained on true words) | true words | 1 | 96.4 (95.0 to 97.7) | 96.4 | 97.6 |
| lilt (word boxes, trained on ocr words) | true words | 3 | 93.2 (90.9 to 95.4) | 93.6, 92.8, 93.0 | 95.9, 96.0, 95.7 |
| lilt (word boxes, trained on ocr words) | OCR words | 3 | 80.5 (76.2 to 84.4) | 80.5, 80.2, 80.8 | 95.9, 96.0, 95.7 |
| lilt (word boxes, trained on ocr words) | true words, one box for all | 3 | 79.0 (75.5 to 82.5) | 77.8, 80.4, 78.8 | 95.9, 96.0, 95.7 |
| lilt (word boxes, trained on true words) | true words | 3 | 93.9 (92.0 to 95.9) | 94.1, 93.7, 94.0 | 95.9, 96.2, 95.9 |
| lilt (word boxes, trained on true words) | OCR words | 3 | 79.8 (75.3 to 83.8) | 79.9, 79.7, 79.8 | 95.9, 96.2, 95.9 |
| lilt (word boxes, trained on true words) | true words, one box for all | 3 | 85.0 (82.0 to 88.0) | 82.1, 89.4, 83.3 | 95.9, 96.2, 95.9 |
| roberta (text only, trained on true words) | true words | 3 | 93.4 (91.1 to 95.7) | 93.2, 93.7, 93.4 | 95.4, 95.5, 95.0 |
| roberta (text only, trained on true words) | OCR words | 3 | 78.9 (74.3 to 83.1) | 78.5, 79.1, 79.1 | 95.4, 95.5, 95.0 |

## Comparisons (paired over receipts)

| A | B | Input | A minus B | p |
|---|---|---|---:|---:|
| lilt (word boxes, trained on true words) | roberta (text only, trained on true words) | true words | 0.5 (-0.5 to 1.5) | 0.327 |
| layoutlmv3 (word boxes, trained on true words) | lilt (word boxes, trained on true words) | true words | -0.2 (-1.4 to 0.9) | 0.673 |
| layoutlmv3 (word boxes, trained on true words) | roberta (text only, trained on true words) | true words | 0.3 (-0.7 to 1.3) | 0.634 |
| lilt (word boxes, trained on true words) | roberta (text only, trained on true words) | OCR words | 0.9 (0.2 to 1.6) | 0.010 |
| layoutlmv3 (word boxes, trained on true words) | lilt (word boxes, trained on true words) | OCR words | 0.5 (-0.3 to 1.2) | 0.222 |
| layoutlmv3 (word boxes, trained on true words) | roberta (text only, trained on true words) | OCR words | 1.3 (0.5 to 2.2) | 0.003 |
| layoutlmv3 (word boxes, trained on true words) | layoutlmv3 (word boxes, trained on true words) | OCR words against true words | -13.5 (-17.4 to -10.2) | < 0.001 |
| layoutlmv3 (word boxes, trained on true words) | layoutlmv3 (word boxes, trained on true words) | true words, one box for all against true words | -35.5 (-38.8 to -32.3) | < 0.001 |
| layoutlmv3 (word boxes, trained on true words) | layoutlmv3 (word boxes, trained on true words) | true words, blank image against true words | 0.2 (-0.1 to 0.4) | 0.260 |
| lilt (word boxes, trained on ocr words) | lilt (word boxes, trained on ocr words) | OCR words against true words | -12.6 (-16.1 to -9.7) | < 0.001 |
| lilt (word boxes, trained on ocr words) | lilt (word boxes, trained on ocr words) | true words, one box for all against true words | -14.2 (-16.7 to -11.6) | < 0.001 |
| lilt (word boxes, trained on ocr words) | lilt (word boxes, trained on true words) | OCR words | 0.7 (-0.2 to 1.8) | 0.127 |
| lilt (word boxes, trained on true words) | lilt (word boxes, trained on true words) | OCR words against true words | -14.2 (-18.1 to -10.6) | < 0.001 |
| lilt (word boxes, trained on true words) | lilt (word boxes, trained on true words) | true words, one box for all against true words | -9.0 (-11.2 to -6.6) | < 0.001 |
| roberta (text only, trained on true words) | roberta (text only, trained on true words) | OCR words against true words | -14.5 (-18.4 to -11.1) | < 0.001 |

## Field F1 by category (true words; fields in the test truth in brackets)

| Category | layoutlmv3 (segment) | layoutlmv3 | lilt (segment) | lilt (OCR trained) | lilt | roberta |
|---|---:|---:|---:|---:|---:|---:|
| menu.nm (251) | 95.9 | 92.0 | 98.4 | 91.1 | 92.3 | 90.4 |
| menu.price (246) | 98.0 | 98.3 | 97.6 | 97.3 | 98.1 | 98.3 |
| menu.cnt (220) | 97.3 | 97.1 | 99.3 | 96.7 | 98.2 | 97.6 |
| total.total_price (96) | 93.2 | 95.3 | 96.3 | 93.1 | 93.8 | 93.1 |
| total.cashprice (69) | 94.8 | 92.6 | 95.6 | 93.8 | 93.1 | 93.9 |
| menu.unitprice (67) | 92.1 | 93.4 | 91.9 | 92.3 | 91.8 | 92.9 |
| sub_total.subtotal_price (66) | 96.4 | 99.2 | 97.8 | 97.5 | 98.8 | 98.3 |
| total.changeprice (56) | 99.1 | 95.0 | 97.3 | 94.8 | 95.6 | 95.6 |
| sub_total.tax_price (45) | 94.5 | 94.8 | 95.6 | 94.8 | 95.2 | 95.2 |
| menu.sub.nm (36) | 84.8 | 87.3 | 91.7 | 81.9 | 86.9 | 83.9 |
| total.menuqty_cnt (29) | 93.5 | 95.6 | 96.7 | 92.7 | 90.7 | 91.0 |
| menu.sub.price (20) | 91.9 | 89.3 | 86.5 | 89.3 | 93.0 | 93.0 |
| menu.sub.cnt (17) | 59.3 | 91.7 | 91.4 | 87.0 | 88.9 | 88.7 |
| total.creditcardprice (17) | 88.9 | 82.8 | 97.1 | 93.3 | 90.2 | 88.2 |
| sub_total.etc (13) | 80.0 | 83.1 | 84.6 | 85.7 | 80.0 | 77.5 |
| sub_total.service_price (12) | 100.0 | 100.0 | 100.0 | 97.2 | 100.0 | 100.0 |
| menu.num (11) | 88.0 | 81.4 | 100.0 | 95.2 | 100.0 | 100.0 |
| menu.discountprice (10) | 88.9 | 58.2 | 94.7 | 66.7 | 53.6 | 57.1 |
| sub_total.discount_price (7) | 100.0 | 100.0 | 100.0 | 100.0 | 95.2 | 95.2 |
| total.menutype_cnt (7) | 66.7 | 78.9 | 83.3 | 73.7 | 76.9 | 76.9 |
| menu.itemsubtotal (6) | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| menu.etc (3) | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| total.total_etc (3) | 0.0 | 20.7 | 33.3 | 26.1 | 28.6 | 27.3 |
| total.emoneyprice (2) | 50.0 | 35.3 | 66.7 | 46.2 | 46.2 | 46.2 |
