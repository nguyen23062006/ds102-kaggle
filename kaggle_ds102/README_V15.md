# V15 — conditional follow-up to V14

Prepared before V14 completes all folds, from the observed rapid drop in training loss (about0.13 at epoch3, near0.01 by epoch8). V14 first-fold blend gain was small; this follow-up is adaptive research, not a fresh independent holdout. Run only after V14 and only if V14 does not pass its screening criteria.

Keep the same randomly initialized TextCNN architecture, train-only vocabulary/scaler, seed102, AdamW lr0.0015, batch64, baselineSVD128 and20% mixture. Change the training regularization schedule to5epochs, dropout0.5, embedding dropout0.2, weight decay0.05; average probability snapshots3/4/5. No pretrained weights/corpus, no dev fitting. All fits hosted Colab, no local training.

Screening unchanged from V14: OOF+0.001, >=2/3 positive fold deltas, worst-fold delta>=-0.004, novel fake F1 delta>=-0.01. Bootstrap intervals reported as uncertainty, not a binary rejection gate. If passed, full train fit and dev guardrail no more than0.005 decline before CSV. Kaggle score unknown until submitted; baseline user-reported score0.77436.

Implement an exact early-pruning rule: when a completed fold has delta<-0.004, stop before fitting the next fold because later folds cannot repair this mandatory gate. Preserve finished fold checkpoints and report only the completed folds; do not compute full OOF using zero-filled missing predictions. This does not change any gate or choose hyperparameters.

Source: scratch_textcnn_v15.py, experiment_v15.py. Model artifacts stay on Colab. This changes the schedule based on training behavior, not selecting an epoch from validation labels. Reused OOF/dev and trying multiple configurations still creates selection bias.
