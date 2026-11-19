# FCA-PAD v7 amendment 1 (written 2026-11-02T08:23:48, after the first runs)
The director limit T used by F13 is now fixed from the original log (an approval policy), not re-learned from the
polluted training data. Reason: in BPIC20_R only about 30 training cases reach the director; the C3/C3x injections remove
the director step from most of them, so the re-learned limit failed its own check and F13 never fired (this also
affected v6). All other thresholds are still learned from the polluted training cases. Added after seeing results:
cell D.6 (share of review slots given to the detector), reported as exploratory.
