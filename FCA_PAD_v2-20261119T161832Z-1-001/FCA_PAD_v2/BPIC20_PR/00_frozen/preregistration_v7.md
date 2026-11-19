# FCA-PAD v7 pre-registration (stress tests)
Written: 2026-11-02T08:23:48   Log: BPIC20_PR
Setup as v6: pollute ALL cases, then the case-level split; detectors (LSTM, GAMA 20 ep, DAE 100 ep)
trained on polluted TRAIN; rule thresholds from polluted TRAIN. Seeds 404, 505, 606 (101 = development only).
Pollution: costly families, each 2% of all cases:
  covered = A1, A5, C1, C2, C3, C4 (each has an audit rule);
  dodge   = A5x partial approval bypass (keeps the first approval, so F11 is silent),
            C1x duplicate claim re-filed 61–120 days later with the amount cut by 1–10% (F12 silent),
            C3x split claims filed 8–20 days apart (F13 silent; logs with director routing only),
            C4x paid after rejection with an approval re-recorded after the rejection (F14 silent);
  new     = N1 approval recorded under the claimant's own role (EMPLOYEE),
            N2 padded amount ×2–4 (kept below the budget p99 where a budget exists, so F15 is silent),
            N3 final approval recorded after the payment.
  Decoys D1–D4: 30% of all cases.
Fitting: evidence model β, λ and the money transform on VAL with labels of COVERED types only
(dodge and new cases removed from the fit). Secondary: all types labelled.
Test pools (TEST split, decoys 10:1 per costly case, all normal cases):
  covered / dodge / new / mixed (half covered, half dodge+new, same size as covered).
Primary metric: MoneyRecall@50 = share of money at stake (loss + exposure) in the top 50.
Hypotheses:
- H23 Rules alone × money recovers much less on dodge and new pools than on the covered pool.
- H24 On dodge, new and mixed pools, detector + layer + exploration recovers more money than rules alone
      (Wilcoxon over detector × log × seed; per-pair 95% bootstrap intervals).
- H25 Rule damage (rules removed at random; thresholds ×[0.5, 0.75, 1.0, 1.5, 2.0]): rules alone loses more
      MoneyRecall than detector + layer + exploration (β refitted, λ and transform kept).
- H26 Learned weights recover at least as much as equal weights; learned rule weights rise with the
      rule's precision on validation (Spearman > 0).
- H27 Mixed pool: detector + layer + exploration beats both rules alone and the detector alone.
