# M6: the service end to end

On Apple M2 with 8 cores, the CPU only, over HTTP on the same machine; every image sent one at a time unless said. The first page took 1.1 s and the first receipt 5.6 s (docTR's detector and LiLT load then).

| Kind | Images | Median wait | 95th percentile | Finding | Reading | Fields | Page CER | Field F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| page | 150 | 1.42 s | 1.95 s | 0.15 s | 1.20 s |  | 1.6% |  |
| receipt | 100 | 1.23 s | 1.46 s | 0.77 s | 0.10 s | 0.35 s | 6.0% | 79.6% (74.8 to 83.8) |

Finding, reading and fields are medians of the service's own times; the wait adds HTTP and decoding the image.

| Clients at once | Receipts a second |
|---:|---:|
| 1 | 0.76 |
| 2 | 0.75 |
| 4 | 0.73 |

Peak memory of the service: 3.30 GB. The models folder: 516 MB.
