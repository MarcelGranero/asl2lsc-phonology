# Labels

`10_data_prep/make_labels.py` derives the label CSVs from the deduped
Sem-Lex metadata. Only `label_type == "asllex"` rows carry phonological
features. Feature order everywhere (CSV columns `f00..f15`, head order) is
the SHuBERT paper's Table 9 row order, recorded in
`results/label_maps.json`.

**NA is a class.** Feature cells are blank when a property is
structurally absent; for all 16 features, observed classes + (1 if blanks
exist) equals the paper's Table 10 class count exactly, so blanks map to
an explicit `__NA__` class (always the last id), trained with plain
cross-entropy and counted in the Recall@1 denominator. `make_labels.py`
hard-asserts the Table 10 counts on every run:

| Feature | classes (incl. NA) | | Feature | classes (incl. NA) |
|---|---:|---|---|---:|
| Major Location | 5 | | Wrist Twist | 2 |
| Minor Location | 37 | | Selected Fingers | 12 |
| Second Minor Location | 37 | | Thumb Position | 2 |
| Contact | 2 | | Flexion | 8 |
| Thumb Contact | 3 | | Spread | 3 |
| Sign Type | 6 | | Spread Change | 3 |
| Repeated Movement | 2 | | Nondominant Handshape | 56 |
| Path Movement | 8 | | Handshape | 58 |

The test split is signer-independent (zero signer overlap with train or
val); the metadata's 17th feature column, `Flexion Change`, is not in
Table 9 and is excluded.
