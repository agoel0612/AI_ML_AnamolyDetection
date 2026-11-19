# FCA-PAD v8 pre-registration (final two-lane design)
Written: 2026-10-20T09:56:04   Log: BPIC20_D
Fresh seeds 707, 808, 909 (never used before). World generator, pollution and split exactly as v7
(covered / dodge / new families at 2% each, decoys 30%, all cases polluted before the split);
GAMA (20 epochs) trained on polluted TRAIN; statistical rule thresholds from polluted TRAIN; the director
limit is a fixed policy constant (v7 amendment 1).
Proposed system (fixed): Lane 1 = learned-weight audit rules (registry v3, F1–F15) × money × (1 + λg), fitted on VAL with
labels of covered types only (β logistic C=1; λ, money transform chosen as in v5); no detector inside.
Lane 2 = GAMA raw score. Review queue of 50: top 40 by Lane 1, then the top 10 by GAMA not yet chosen.
Comparators: rules alone × money (Lane 1 only, 50 slots); GAMA alone; GAMA + layer (blended, v7);
GAMA + layer + exploration (v7 hybrid); two-lane with equal-weight rules; rules with equal weights × money; amount alone.
Test pools on TEST (decoys 10:1): covered / dodge / new / mixed (half covered, half dodge + new).
Metric: MoneyRecall@50 (share of loss + exposure in the top 50); also share of costly cases, share of
rule-less costly cases (dodge + new) in the top 50.
Tests: one-sided Wilcoxon signed-rank over log × seed (4 × 3 = 12 pairs), alpha 0.05; non-inferiority margin 0.05.
Primary hypotheses (mixed pool):
- H28 Two-lane recovers more money than the blended GAMA + layer.
- H29 Two-lane is non-inferior to rules alone (Δ > −0.05) and catches more rule-less costly cases.
Secondary:
- H30 Dodge and new pools: two-lane recovers more money than rules alone.
- H31 Covered pool: two-lane is non-inferior to rules alone (Δ > −0.05).
- H32 Covered and mixed pools: two-lane recovers more money than GAMA alone.
- H33 Two-lane with learned rule weights recovers more than with equal weights (covered, mixed).
- H34 Removing 0→10 of 13 rules at random (10 draws): two-lane loses less MoneyRecall than rules alone.
