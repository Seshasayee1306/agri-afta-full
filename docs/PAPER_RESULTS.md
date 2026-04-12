# Paper Results Summary

## Evaluation Setup
- Dataset: `dataset/irrigation_stage_dataset.csv`
- Test strategy: 3 holdout protocols (`random`, `time`, `client`) with 20% test split
- Models compared: AFTA (`backend/final_model.pkl`), Stage RF (`backend/stage_models/*.pkl`), OR-ensemble
- Metrics: accuracy, balanced accuracy, precision, recall, F1, and bootstrap 95% CI for accuracy/F1

## Headline Result (AFTA)
- Mean accuracy across splits: **0.9739**
- Mean balanced accuracy across splits: **0.9718**
- Mean F1 across splits: **0.9660**

## Per-Split Results
| split | model | rows | accuracy | balanced_accuracy | precision | recall | f1 | acc_ci95_low | acc_ci95_high | f1_ci95_low | f1_ci95_high |
|---|---|---|---|---|---|---|---|---|---|---|---|
| random | AFTA | 20000 | 0.9747 | 0.9727 | 0.9701 | 0.9640 | 0.9670 | 0.9724 | 0.9769 | 0.9639 | 0.9699 |
| random | StageRF | 20000 | 0.6016 | 0.5433 | 0.4715 | 0.2899 | 0.3590 | 0.5947 | 0.6079 | 0.3476 | 0.3697 |
| random | EnsembleOR | 20000 | 0.8555 | 0.8778 | 0.7356 | 0.9748 | 0.8385 | 0.8502 | 0.8601 | 0.8325 | 0.8438 |
| time | AFTA | 20000 | 0.9742 | 0.9721 | 0.9702 | 0.9627 | 0.9664 | 0.9718 | 0.9765 | 0.9638 | 0.9698 |
| time | StageRF | 20000 | 0.6008 | 0.5429 | 0.4715 | 0.2898 | 0.3590 | 0.5941 | 0.6071 | 0.3471 | 0.3698 |
| time | EnsembleOR | 20000 | 0.8552 | 0.8774 | 0.7359 | 0.9745 | 0.8385 | 0.8503 | 0.8600 | 0.8327 | 0.8443 |
| client | AFTA | 20082 | 0.9727 | 0.9707 | 0.9672 | 0.9621 | 0.9646 | 0.9704 | 0.9748 | 0.9618 | 0.9673 |
| client | StageRF | 20082 | 0.6010 | 0.5435 | 0.4755 | 0.2878 | 0.3585 | 0.5939 | 0.6079 | 0.3491 | 0.3680 |
| client | EnsembleOR | 20082 | 0.8562 | 0.8780 | 0.7381 | 0.9747 | 0.8401 | 0.8509 | 0.8613 | 0.8345 | 0.8455 |

## Interpretation
- AFTA is consistently strong and stable across random/time/client splits.
- StageRF alone underperforms AFTA; OR-ensemble increases recall but reduces precision/accuracy.
- For deployment and paper claims on reliability, AFTA standalone is the primary model.
- OR-ensemble can be positioned as a high-recall safety mode when missed irrigation is costly.