# M5: DocLens against other OCR systems

Every system read the same images (scripts/m5_export.py) through its own runner (scripts/m5/); scripts/m5_score.py scored them. Intervals are bootstrap 95% intervals; each system is paired with DocLens on the same images (points of CER, or of F1).

## Systems

| System | Version | Runs on | Licence |
|---|---|---|---|
| DocLens | crnn 025cd01f20acb2f5 code b73b86ded226bedf greedy batch 64 | mps | MIT (docTR's detector, used for receipts: Apache 2.0) |
| Tesseract | tesseract 5.5.3 best psm 7 | cpu | Apache 2.0 |
| PaddleOCR | paddleocr 3.7.0 defaults for the language, no preprocessing | cpu | Apache 2.0 (code and models) |
| EasyOCR | easyocr 1.7.2 default settings | mps | Apache 2.0 |
| PaddleOCR VL 1.6 | paddleocr 3.7.0 PaddlePaddle/PaddleOCR-VL-1.6 mlx_vlm 0.7.11, HTML as text | layout on the CPU, the model on the Apple GPU through MLX VLM | Apache 2.0 (code and the 0.96 billion parameter model) |
| Surya 2 | surya 0.22.1 surya-2.gguf full page | Apple GPU through llama.cpp | code Apache 2.0; weights AI PUBS OpenRAIL M, modified: no use by an organisation above $5 million of revenue or funding, or in a competing product, except personal use and research |

## Crops: Mozhi words (Hindi) (1000 images, CER)

| System | CER | Against DocLens | Read perfectly | CER above 50% | Images per second |
|---|---:|---:|---:|---:|---:|
| DocLens | 0.5% (0.2 to 0.7) |  | 98.5% | 0.0% | 316.8 |
| Tesseract | 6.5% (5.4 to 7.9) | 6.1 more (p < 0.001) | 84.3% | 8.9% | 389.3 |
| PaddleOCR | 5.2% (4.4 to 6.1) | 4.8 more (p < 0.001) | 84.6% | 6.0% | 9.9 |
| EasyOCR | 6.8% (5.8 to 7.8) | 6.3 more (p < 0.001) | 80.3% | 6.3% | 36.0 |
| PaddleOCR VL 1.6 | 56.6% (50.6 to 63.7) | 56.2 more (p < 0.001) | 35.3% | 50.6% | 2.2 |
| Surya 2 | 20.2% (15.6 to 25.4) | 19.8 more (p < 0.001) | 78.7% | 13.0% | 3.6 |

## Crops: Synthetic lines, unseen fonts (1000 images, CER)

| System | CER | Against DocLens | Read perfectly | CER above 50% | Images per second |
|---|---:|---:|---:|---:|---:|
| DocLens | 1.0% (0.8 to 1.2) |  | 83.1% | 0.0% | 82.0 |
| Tesseract | 3.1% (2.7 to 3.5) | 2.1 more (p < 0.001) | 64.5% | 0.0% | 34.1 |
| PaddleOCR | 7.0% (6.1 to 8.0) | 6.0 more (p < 0.001) | 46.9% | 2.5% | 4.5 |
| EasyOCR | 3.5% (3.2 to 3.8) | 2.5 more (p < 0.001) | 45.7% | 0.0% | 13.5 |
| PaddleOCR VL 1.6 | 13.9% (12.2 to 15.8) | 13.0 more (p < 0.001) | 35.2% | 7.0% | 1.2 |
| Surya 2 | 0.7% (0.6 to 0.9) | 0.3 fewer (p 0.027) | 84.5% | 0.0% | 2.0 |

## Crops: CORD words (1000 images, CER)

| System | CER | Against DocLens | Read perfectly | CER above 50% | Images per second |
|---|---:|---:|---:|---:|---:|
| DocLens | 1.2% (0.9 to 1.6) |  | 95.4% | 0.5% | 297.1 |
| Tesseract | 20.6% (18.3 to 22.9) | 19.3 more (p < 0.001) | 63.2% | 17.8% | 367.8 |
| PaddleOCR | 1.4% (1.1 to 1.8) | 0.2 more (p 0.376) | 93.9% | 1.0% | 7.4 |
| EasyOCR | 17.5% (15.9 to 19.4) | 16.3 more (p < 0.001) | 59.4% | 11.4% | 100.5 |
| PaddleOCR VL 1.6 | 11.7% (9.4 to 14.5) | 10.4 more (p < 0.001) | 75.5% | 7.2% | 2.0 |
| Surya 2 | 66.6% (38.9 to 102.4) | 65.4 more (p < 0.001) | 66.7% | 20.8% | 1.7 |

## Crops: CORD lines (791 images, CER)

| System | CER | Against DocLens | Read perfectly | CER above 50% | Images per second |
|---|---:|---:|---:|---:|---:|
| DocLens | 2.8% (2.3 to 3.3) |  | 78.8% | 0.1% | 89.5 |
| Tesseract | 42.8% (39.5 to 46.1) | 40.0 more (p < 0.001) | 28.8% | 37.4% | 162.6 |
| PaddleOCR | 8.6% (7.8 to 9.4) | 5.8 more (p < 0.001) | 26.0% | 1.8% | 3.8 |
| EasyOCR | 21.9% (20.1 to 23.8) | 19.2 more (p < 0.001) | 27.2% | 12.5% | 54.7 |
| PaddleOCR VL 1.6 | 16.0% (13.6 to 18.7) | 13.2 more (p < 0.001) | 57.3% | 10.7% | 1.6 |
| Surya 2 | 41.0% (33.7 to 50.6) | 38.2 more (p < 0.001) | 49.9% | 25.2% | 1.6 |

## Pages: Synthetic pages, unseen fonts (150 pages, page CER, intervals over 30 page designs)

| System | Page CER | Against DocLens | Median page | Pages above 50% | Seconds per page |
|---|---:|---:|---:|---:|---:|
| DocLens | 1.6% (0.6 to 2.8) |  | 0.3% | 0.7% | 1.09 |
| Tesseract | 9.7% (7.2 to 12.1) | 8.1 more (p < 0.001) | 2.0% | 2.0% | 1.68 |
| PaddleOCR | 40.5% (26.7 to 53.3) | 38.9 more (p < 0.001) | 36.9% | 50.0% | 28.87 |
| EasyOCR | 48.2% (36.9 to 58.6) | 46.6 more (p < 0.001) | 71.0% | 56.0% | 20.82 |
| PaddleOCR VL 1.6 | 37.5% (24.0 to 51.6) | 35.9 more (p < 0.001) | 11.2% | 14.7% | 40.69 |
| Surya 2 | 0.4% (0.3 to 0.5) | 1.1 fewer (p 0.006) | 0.3% | 0.0% | 33.83 |

| System | one column | two columns | clean | noise | photo | shadow | skew |
|---|---:|---:|---:|---:|---:|---:|---:|
| DocLens | 0.5% | 2.5% | 0.7% | 1.0% | 4.7% | 1.0% | 0.5% |
| Tesseract | 12.8% | 6.8% | 1.6% | 2.7% | 19.2% | 15.5% | 9.3% |
| PaddleOCR | 1.6% | 75.1% | 40.3% | 40.5% | 40.6% | 40.3% | 40.8% |
| EasyOCR | 16.7% | 76.2% | 42.9% | 42.9% | 54.6% | 42.7% | 57.9% |
| PaddleOCR VL 1.6 | 38.7% | 36.4% | 10.3% | 38.5% | 62.2% | 12.1% | 64.4% |
| Surya 2 | 0.4% | 0.4% | 0.4% | 0.4% | 0.4% | 0.4% | 0.5% |

## Pages: CORD receipts (100 pages, page CER, intervals over 100 receipts)

| System | Page CER | Against DocLens | Median page | Pages above 50% | Seconds per page |
|---|---:|---:|---:|---:|---:|
| DocLens | 6.0% (4.0 to 8.6) |  | 1.8% | 1.0% | 0.37 |
| Tesseract | 51.8% (44.6 to 59.5) | 45.7 more (p < 0.001) | 49.4% | 50.0% | 0.08 |
| PaddleOCR | 11.5% (9.1 to 14.2) | 5.4 more (p 0.001) | 8.2% | 1.0% | 5.21 |
| EasyOCR | 25.9% (22.0 to 30.4) | 19.9 more (p < 0.001) | 20.3% | 10.0% | 2.52 |
| PaddleOCR VL 1.6 | 8.3% (5.6 to 11.8) | 2.2 more (p 0.228) | 2.4% | 3.0% | 4.13 |
| Surya 2 | 13.7% (6.7 to 24.7) | 7.7 more (p < 0.001) | 1.9% | 7.0% | 13.74 |

## PaddleOCR VL on the Mac and in its reference implementation

The same crops read by PaddleOCR VL's pipeline, its model served by MLX VLM as its Apple silicon guide sets it up, and by the model in Hugging Face Transformers in float32 on the CPU (decision 48).

| Set | Crops | Pipeline, MLX VLM | Transformers, CPU | Pipeline against Transformers | Seconds per crop, Transformers |
|---|---:|---:|---:|---:|---:|
| Mozhi words (Hindi) | 200 | 48.8% (41.6 to 56.3) | 42.0% (33.6 to 51.7) | 6.9 more (p 0.079) | 1.74 |
| Synthetic lines, unseen fonts | 200 | 14.7% (11.5 to 18.6) | 11.2% (8.9 to 13.7) | 3.5 more (p 0.027) | 4.92 |
| CORD lines | 100 | 14.1% (8.9 to 19.5) | 12.1% (7.6 to 17.6) | 1.9 more (p 0.400) | 3.24 |

## Fields on receipts (100 receipts, field F1, seeds pooled: 3 RoBERTa, 3 LiLT)

| System | RoBERTa, the system's own order | Against DocLens | LiLT, words and boxes in rows | Against DocLens | Perfect tags on these words |
|---|---:|---:|---:|---:|---:|
| DocLens | 78.9% (74.3 to 83.0) |  | 79.7% (75.3 to 83.8) |  | 85.3% |
| Tesseract | 32.1% (25.5 to 38.6) | 46.8 below (p < 0.001) | 38.0% (31.8 to 44.4) | 41.7 below (p < 0.001) | 44.5% |
| PaddleOCR | 73.7% (69.6 to 77.5) | 5.2 below (p 0.009) | 82.4% (79.0 to 85.7) | 2.7 above (p 0.058) | 84.2% |
| EasyOCR | 29.2% (24.7 to 33.9) | 49.7 below (p < 0.001) | 31.5% (26.7 to 36.4) | 48.2 below (p < 0.001) | 37.3% |
| PaddleOCR VL 1.6 | 79.4% (74.3 to 83.8) | 0.5 above (p 0.803) | 78.0% (73.8 to 81.9) | 1.8 below (p 0.356) | 80.7% |
| Surya 2 | 78.2% (72.7 to 83.0) | 0.7 below (p 0.817) | no boxes for lines |  |  |

## Memory and loading

| System | Peak memory, crops | Peak memory, pages | Loading, crops |
|---|---:|---:|---:|
| DocLens | 4.35 GB | 4.91 GB | 0 s |
| Tesseract | 0.15 GB | 0.22 GB | 0 s |
| PaddleOCR | 2.18 GB | 18.33 GB | 2 s |
| EasyOCR | 3.39 GB | 21.20 GB | 8 s |
| PaddleOCR VL 1.6 | 8.42 GB | 11.75 GB | 2 s |
| Surya 2 | 11.86 GB | 14.85 GB | 3 s |

Memory is the peak physical footprint (macOS), which on Apple silicon includes what the GPU holds for the process, plus the peak of the server a system runs its model in (Surya: llama.cpp; PaddleOCR VL: MLX VLM). Tesseract's is the largest single process; it runs one per core.
