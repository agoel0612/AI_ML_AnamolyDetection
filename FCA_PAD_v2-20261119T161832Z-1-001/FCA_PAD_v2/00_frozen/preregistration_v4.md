# FCA-PAD v4 pre-registration (realistic anomaly mix)
Written: 2026-10-13T04:28:52   Log: BPIC20_D

Worlds: new injection seeds [404, 505, 606], val/test only, training untouched.
Costly anomalies: 3% of val/test cases, equal share per available type
(A1, A5, C1, C2, C3, C4; C2 only with a budget field, C3 only with director routing).
Decoys: 10 per costly anomaly, equal share of D1, D2, D3, D4.
Money: A1, C1 = case amount paid twice (loss); C2 = inflated part (loss); C4 = amount paid
despite rejection (loss); A5, C3 = amount paid without the required approval (exposure).

Positives for detection metrics: costly anomalies only; decoys count as negatives.
Selection: on VAL of each world (all decoys), objective ⅓ PR-AUC + ⅔ Loss Recall@50,
PR-AUC floor −0.02; λ ∈ [0, 0.25, 0.5, 1, 2, 4, 8]; money scale ∈ {log, linear}.

Layers compared: v3 form  = P(anomaly) × money × (1 + λ·g)
                 v4 form  = P(costly | behaviour z, rule hits) × money × (1 + λ·g), where
                            P(costly | ·) is a logistic model fitted on VAL (rules act as evidence,
                            so a rule can surface an anomaly the process model cannot see).
Primary system: v4 form with the FROZEN registry v1.
Primary hypotheses (TEST, decoy ratio 10:1, mean over seeds, paired-bootstrap 95% CIs):
- H10 The primary system recovers more lost money in the top 50
      than P(anomaly) alone and than the v2 fusion.
- H11 It does so on hold-out types with no v1 rule (A5, C1–C4).
- H12 Its benefit grows with the decoy ratio (0, 1, 4, 10).
- H13 Net benefit = lost money recovered in top 50 − review cost × 50 is higher than for
      P(anomaly) alone at every review cost in [10, 25, 50, 100] €/case.
Secondary: registry v3 audit tests (F11–F15) add further recovery; their firing rate on
natural TRAIN cases is reported as the false-alarm cost.
Registry v3 rules (author-defined from standard audit tests, fixed here):
- F11 payment requested or made with no approval recorded
- F12 same rare amount (≤ 3 TRAIN occurrences) and same organisation key in an earlier case
      within 60 days (duplicate claim)
- F13 two cases of the same organisation within 7 days, each below the director
      threshold T, together at or above T (threshold splitting)
- F14 paid or requested after a rejection with no approval since that rejection
- F15 amount / budget above the TRAIN 99th percentile of that ratio
