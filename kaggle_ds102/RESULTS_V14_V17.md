# Research V14–V17 — best-known Kaggle 0.77436 retained

The user reported submission_v11_experimental.csv scored0.77436. This updates the known-best baseline from v6=0.77269. Historical reports are retained; current SVD128 baseline OOF macroF1 is0.7749451431.

## V14: randomly initialized TextCNN, 12 epochs

Executed two full CV folds on hosted Colab CPU. CNN alone macroF1: fold1=0.7234217165, fold2=0.6906913176. Fixed20%CNN blend versus SVD128: fold1 delta+0.0001833536, fold2 delta-0.0085278902. The mandatory worst-fold floor was-0.004. Stopped the background process after this irreversible violation, while fold3 was preparing features; process exit-15 is intentional pruning, not an unexplained crash. Completed fold checkpoints retained. No full3fold OOF score is claimed.

CNN architecture is inspired by CNN-rand, with additional product/title/numeric context, all weights/vocabulary learned from scratch from the fit partition. Source reference: https://aclanthology.org/D14-1181/ . This is not a reproduction of that paper's results; no pretrained vectors/checkpoints or outside corpus were used. PyTorch2.11.0+cpu on Colab, BSD-3-Clause.

## V15: short schedule and stronger regularization

Five epochs, snapshot average epochs3/4/5, dropout0.5, embedding dropout0.2, AdamW decay0.05. Two full folds on Colab. CNN-alone F1 fold1=0.7269809077, fold2=0.7116554915; blend delta fold1=-0.0000513894, fold2=-0.0132149380. Automatically pruned on the same fixed worst-fold floor; exit0. Better standalone CNN on these folds did not translate to a better blend. No full3fold OOF score claimed.

## V16: binary fake-review expert using copy-alignment features

Added row-local fuzzy substring/diacritic-normalized/long contiguous word block features versus product title/description. No hard rule assigns labels; binary CatBoost learns fake versus other classes from the training fold. Fixed15%expert blend. Fold1 delta0, fold2 delta-0.0044889830: automatically pruned under the same floor, exit0. No complete OOF score or claim of improvement. Feature label-independence and probability-conservation checks passed on Colab. RapidFuzz3.14.6 MIT; metrics documentation https://rapidfuzz.github.io/RapidFuzz/Usage/fuzz.html . Long-string partial matching is heuristic.

## V17: decision-weight recalibration for the new 128D representation

Reused the existing coarse selection_v5.fit_weights grid unchanged. Calibrator fit on two train-OOF folds and evaluated on the third. Fixed old weights baseline:0.7749451431; cross-fitted recalibration:0.7725163864, delta-0.0024287567. Fold deltas -0.0005667251/-0.0079499221/+0.0015654755. Proposed full weights[1,1.2,1.4,0.9], but its in-sample fit score0.7757306187 is not validation and was not used to override the failed cross-fit assessment. Novel fakeF1 delta-0.0272464566. All screening gates failed; no CSV exported.

## Outcome and limits

No new recommended submission from these four trials. Keep the SVD128 file that actually scored0.77436 on Kaggle. None of these negative results proves no neural model or copy feature can work; these fixed implementations/configurations did not improve under the stated comparisons.

All ML calculations/training took place on hosted Colab. Only source packaging/AST/file operations on Mac. No pretrained models, external training data, changed labels or dev/test fitting. Early pruning avoided wasting computation after hard gates became impossible. Bootstrap significance was not a binary gate in these new trials, as recorded before outcomes; completed-fold/gain/novel-fake safeguards still applied. Repeated adaptive OOF/dev use causes selection bias; calibration cross-fitting is not fully nested model CV.

Report/manifest files are saved in output_v14_experiment through output_v17_experiment. CNN state_dict snapshots and features remain only on Colab. No automatic Kaggle submission occurred. All trial processes have finished or were explicitly pruned.

## Conditional next step, not run

Kaggle Rules/Data pages were inaccessible to the web tool. Asked the user whether course rules allow including1590labeled dev rows in final training after model selection. Until confirmed, dev remains separate. Prepared train_final_train_dev.py but did NOT execute it. If allowed, it refits the fixed best SVD128 recipe on train+dev, predicts test only, and reports no held-out dev metric after merging. Source/provenance is ready for review; confirmation must be factual, not inferred from elapsed time or silence.

The user replied that the train+dev rule is unknown. Dev remains held out and the conditional refit was not executed. Final Colab process check returned no active trial or train+dev process.
