# M6: a receipt's sums against the OCR's misreadings

CORD's 100 test receipts, read as in M4 (docTR's detector, the CRNN).

| Rule | Receipts it applies to (true fields) | Holds on the true fields |
|---|---:|---:|
| items add up to the subtotal | 64 | 61 |
| subtotal and its extras give the total | 63 | 57 |
| items add up to the total | 31 | 30 |
| cash less the total is the change | 50 | 47 |

## Perfect tags on the OCR words (only the reading is wrong)

Receipts flagged, a rule failing on the fields found: 22; of them with a misread or lost amount: 17. Receipts with a misread or lost amount: 53; of them flagged: 17.

One character fixes made: 6; right: 2. Field F1 before 85.2%, after 85.1% (difference -0.08 points, -0.41 to +0.24, p 0.788).

| Rule | Field | Read | Fixed to | Right |
|---|---|---|---|---|
| cash less the total is the change | total.total_price | GRAND TOTAL 51,3000 | GRAND TOTAL 51,300 | no |
| cash less the total is the change | total.cashprice | CASH 200,000 | CASH 300,000 | no |
| items add up to the subtotal | menu.price | 165,400 | 165,460 | yes |
| items add up to the subtotal | menu.price | 1108,000 | 108,000 | yes |
| cash less the total is the change | total.changeprice | Kembalian 22000 | Kembalian 2200 | no |
| subtotal and its extras give the total | sub_total.tax_price | Pb1 2,681 | Pb1 2,682 | no |

## LiLT's fields from the OCR words

Receipts flagged, a rule failing on the fields found (summed over 3 seeds): 73; of them with a misread or lost amount: 62. Receipts with a misread or lost amount: 182; of them flagged: 62.

One character fixes made: 19; right: 6. Field F1 before 79.8%, after 79.7% (difference -0.05 points, -0.36 to +0.25, p 0.810).

| Rule | Field | Read | Fixed to | Right |
|---|---|---|---|---|
| cash less the total is the change | total.total_price | GRAND TOTAL 51,3000 | GRAND TOTAL 51,300 | no |
| cash less the total is the change | total.cashprice | CASH 200,000 | CASH 300,000 | no |
| items add up to the subtotal | menu.price | 165,400 | 165,460 | yes |
| items add up to the subtotal | menu.price | 1108,000 | 108,000 | yes |
| subtotal and its extras give the total | sub_total.tax_price | Pb1 2,681 | Pb1 2,682 | no |
| items add up to the total | menu.price | 120,000 | 120,100 | no |
| cash less the total is the change | total.total_price | GRAND TOTAL 51,3000 | GRAND TOTAL 51,300 | no |
| cash less the total is the change | total.cashprice | CASH 200,000 | CASH 300,000 | no |
| items add up to the subtotal | menu.price | 165,400 | 165,460 | yes |
| items add up to the subtotal | menu.price | 1108,000 | 108,000 | yes |
| cash less the total is the change | total.changeprice | Kembalian 22000 | Kembalian 2200 | no |
| subtotal and its extras give the total | sub_total.tax_price | Pb1 2,681 | Pb1 2,682 | no |
