# M4: fields the OCR words lose

CORD's 100 test receipts, 1309 true fields. Given perfect tags (the field of the true words under each OCR word), the OCR words still lose 208 fields (15.9%): numbers 64 of 552 (11.6%), other fields 144 of 757 (19.0%).

| How the closest field of the same category differs | Fields | Share |
|---|---:|---:|
| one character wrong | 116 | 56% |
| two or three characters wrong | 44 | 21% |
| more characters wrong, or joined or split | 39 | 19% |
| no field of its category found | 9 | 4% |

Of the 116 fields one character off, 70 differ in a digit or a separator (. , :).

| Category | Fields lost |
|---|---:|
| menu.nm | 37 |
| menu.cnt | 24 |
| total.total_price | 23 |
| menu.price | 23 |
| total.changeprice | 16 |
| total.cashprice | 12 |
| menu.sub.price | 11 |
| menu.sub.nm | 8 |
| sub_total.subtotal_price | 8 |
| sub_total.tax_price | 8 |
| total.menuqty_cnt | 6 |
| sub_total.etc | 6 |
| total.creditcardprice | 4 |
| menu.unitprice | 4 |
| menu.num | 4 |
| menu.discountprice | 4 |
| menu.itemsubtotal | 3 |
| sub_total.discount_price | 3 |
| total.menutype_cnt | 2 |
| total.total_etc | 1 |
| sub_total.service_price | 1 |

One character apart (true, read): 60.000 as 80.000; EDCCIMBNIAGANo:xx773060.000 as EDCCIMBNIAGANo:xX773060.000; SUBTOTAL28,000 as SUBTOTA28,000; TOTALITEMS1 as TOTALITEMS; 2 as 1; CHANGE25,400 as CHANGE28,400; REDBEANBREAD as REDBEANBIEAD; CHANGE4,000 as CHANCE4,000; 100,000 as 100.000; CashRp.111,000 as CashRp.101,000
