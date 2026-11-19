# FCA-PAD v6 pre-registration (polluted log, then split, retrain, test)
Written: 2026-11-02T06:29:09   Log: BPIC20_PR
Pollution: the v4 generator applied to ALL cases (3% costly, 10× decoys), seeds [404, 505, 606].
Split after pollution: train/val/test (0.6, 0.2, 0.2), stratified by anomaly type, seeded.
Detectors retrained on the polluted TRAIN part without labels: LSTM (max_full), GAMA (20 epochs),
DAE (100 epochs). Same inputs, defaults, calibration and financial layer as v5.
Hypotheses (as v5, now with polluted training):
- H14p Detectors alone catch a small share of lost money in the top 50.
- H15p Detector + layer (rules v3) recovers more lost money than the detector alone (Wilcoxon over detector × log × seed).
- H16p The combination vs rules v3 alone and detector × amount.
- H17p Leave-one-type-out as in v5.
- H18p Net benefit higher than the detector alone at every review cost.
- H20 Pollution effect: change in each system's Loss Recall@50 relative to v5 (clean training), reported per log.
