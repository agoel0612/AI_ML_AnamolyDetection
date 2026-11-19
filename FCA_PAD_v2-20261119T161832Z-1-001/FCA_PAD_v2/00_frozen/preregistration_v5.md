# FCA-PAD v5 pre-registration (benchmark detectors + financial layer)
Written: 2026-10-13T04:40:24   Log: BPIC20_D
Worlds: the v4 generator, seeds [404, 505, 606] (identical worlds to v4).
Detectors: LSTM (variant max_full, fixed), GAMA (20 epochs), DAE (100 epochs, batch 64).
Same footing: TRAIN cases only; inputs activity, role, time-gap band; defaults; no tuning on VAL.
Calibration (all detectors): robust z vs TRAIN cases in the same length bucket; Platt on VAL (any anomaly).
Systems per detector: alone; × amount; + layer (rules v1); + layer (rules v3 audit tests).
Detector-free baselines: amount alone; rules v3 alone × money.
Primary comparison: detector + layer (rules v3) vs detector alone, lost money in top 50, decoys 10:1.
Hypotheses:
- H14 Detectors alone catch a small share of lost money in the top 50 even when they detect anomalies well (AP over all anomalies).
- H15 For every detector, + layer (rules v3) recovers more lost money than the detector alone
      (Wilcoxon signed-rank over detector × log × seed; per-pair 95% bootstrap intervals).
- H16 The combination beats rules v3 alone and detector × amount.
- H17 Leave-one-type-out: with a type's rule removed and its labels withheld from the evidence model,
      detector + layer still catches more of that type than rules alone.
- H18 Net benefit (lost money in top 50 − cost × 50) is higher than the detector alone at every cost in [10, 25, 50, 100] €.
- H19 Hybrid queue: 20% of the 50 slots go to the detector's most anomalous cases not already chosen
      by the layer. It keeps most of the layer's recovery and catches more held-out types (leave-one-type-out).
