# M5: why the systems failed where they did

From what each runner wrote; no model is run (scripts/m5_diagnose.py).

## Two column pages (75 pages): order or reading?

Page CER as each system gives the page, and with its lines or blocks read column after column (a diagnosis: it uses where the columns are).

| System | Its own order | Column after column |
|---|---:|---:|
| DocLens | 2.5% | 2.9% |
| Tesseract | 6.8% | 12.5% |
| PaddleOCR | 75.1% | 2.0% |
| EasyOCR | 76.2% | 5.7% |
| PaddleOCR VL 1.6 | 36.4% | 36.7% |
| Surya 2 | 0.4% | 0.9% |

## Text that is not there

Crops read more than twice as long as the truth (and 5 characters longer), and the share of the system's edits on the set they cause.

| System | mozhi | synth | cord_word | cord_line |
|---|---:|---:|---:|---:|
| DocLens | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) |
| Tesseract | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) |
| PaddleOCR | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) |
| EasyOCR | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) | 2 (1.5%) |
| PaddleOCR VL 1.6 | 19 (19.2%) | 12 (12.2%) | 19 (33.6%) | 21 (25.8%) |
| Surya 2 | 23 (45.8%) | 0 (0.0%) | 43 (78.4%) | 84 (58.4%) |

Pages read more than one and a half times as long as the truth:

* synth: DocLens 0 of 150; Tesseract 0 of 150; PaddleOCR 0 of 150; EasyOCR 0 of 150; PaddleOCR VL 1.6 14 of 150; Surya 2 0 of 150
* cord: DocLens 1 of 100; Tesseract 2 of 100; PaddleOCR 0 of 100; EasyOCR 1 of 100; PaddleOCR VL 1.6 2 of 100; Surya 2 3 of 100

## Script

| System | Mozhi words partly in another script | Synthetic lines partly in another script | Mozhi's lone dandas misread |
|---|---:|---:|---:|
| DocLens | 0.0% | 0.0% | 0 of 52 |
| Tesseract | 0.0% | 4.6% | 48 of 52 |
| PaddleOCR | 0.0% | 0.0% | 51 of 52 |
| EasyOCR | 0.0% | 1.5% | 52 of 52 |
| PaddleOCR VL 1.6 | 9.9% | 2.2% | 52 of 52 |
| Surya 2 | 0.7% | 0.0% | 52 of 52 |
