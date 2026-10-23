# PopSign under the SHuBERT paper protocol — caveats

The `popsign16_*` splits run the exact Sem-Lex trainer; the corpus is
selected by the split-name prefix. These properties of the data bound what
the PopSign numbers mean:

1. **Labels are type-level.** PopSign features are joined from ASL-LEX 2.0
   by gloss, so every clip of a gloss shares one feature vector (250
   distinct vectors). The 16 heads can in principle be satisfied by gloss
   recognition, so high in-domain feature recall is close to tautological.
   Sem-Lex labels are also ASL-LEX-derived, but over a 1,914-sign
   vocabulary with signer-disjoint test — the shortcut is far weaker there.
2. **Head sizes differ from the paper's Table 10** (PopSign's 250 glosses
   do not exercise the full ASL-LEX inventory): per-head recall is not
   comparable head-for-head with the Sem-Lex arm; only within-corpus
   comparisons are meaningful.
3. **NA is a class; −1 is unknown.** The released configuration
   (`popsign16nac_*`) trains structurally-absent values as a real class,
   like Sem-Lex; `-1` marks only genuinely unknown supervision (a gloss
   with no ASL-LEX match) and is excluded from that head's denominator.
4. **Long clips.** PopSign clips reach 1280 frames and batches pad to the
   longest clip, so the protocol batch size needs a large-memory GPU; the
   released run keeps the paper's batch 128.
5. **LSC transfer coverage.** The PopSign Handshape head has fewer classes
   than Sem-Lex's, so part of the LSC inventory can be unreachable by
   construction — read transfer accuracy together with the snap distances
   in the per-clip output.
