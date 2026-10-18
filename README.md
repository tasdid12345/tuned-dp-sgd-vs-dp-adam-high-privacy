# Tuned DP-SGD with momentum vs default DP-Adam at small epsilon

## Motivation
The paper "Adaptive Methods Are Preferable in High Privacy Settings: An SDE Perspective" argues, via a stochastic differential equation analysis, that adaptive optimizers hold an advantage over non-adaptive ones when differential-privacy noise is large. This small experiment probes that claim empirically on a public tabular task: is the adaptive edge real once the non-adaptive method is carefully tuned?

## Question
Does a carefully tuned non-adaptive DP-SGD (with momentum) match or beat a default-setting DP-Adam at the same high-privacy budgets (epsilon in {0.5, 1, 2}, delta=1e-5)?

## Dataset
UCI Adult income, fetched from OpenML (data id 1590). It is not the paper's own dataset. It is one-hot encoded and standardised, then split once (stratified) into 60% train, 20% validation and 20% test.

## Baseline vs improved
- Baseline: numpy logistic regression trained with DP-Adam at common defaults (lr 0.01, clip 1.0).
- Improved: the same model trained with DP-SGD with momentum, with learning rate and clip norm picked from a 5x4 grid on the validation set.
- Reference: DP-Adam tuned on the same kind of grid.
All arms use per-example clipping, Gaussian noise, Poisson batches, and the same steps and batch size. A simple RDP accountant sets the noise multiplier for each epsilon. Hyperparameter-search privacy cost is not accounted for. The main metric is test accuracy averaged over epsilons and 5 noise seeds; test ROC-AUC is secondary.

## How to run
```
pip install -r requirements.txt
python analysis.py
```

## Results
Results are written to `results.json` (and a chart to `figures/`).

## Results (latest run)

On the public UCI Adult test set, default-setting DP-Adam reached 84.9% accuracy versus 85.4% for tuned DP-SGD with momentum (mean over ε∈{0.5,1,2}); tuned DP-Adam also reached 85.4%, so the gain reflects tuning rather than the optimizer.

| Metric | Value |
|---|---|
| baseline_score | 0.8487 |
| baseline_score_std | 0.0007817 |
| baseline_accuracy | 0.8487 |
| baseline_roc_auc | 0.9 |
| baseline_acc_eps0.5 | 0.8471 |
| baseline_acc_eps1.0 | 0.849 |
| baseline_acc_eps2.0 | 0.8499 |
| improved_score | 0.8538 |
| improved_score_std | 0.0007014 |
| improved_accuracy | 0.8538 |
| improved_roc_auc | 0.9028 |
| improved_acc_eps0.5 | 0.853 |
| improved_acc_eps1.0 | 0.8545 |
| improved_acc_eps2.0 | 0.8541 |
| tuned_adam_accuracy | 0.8537 |
| tuned_adam_roc_auc | 0.9026 |
| tuned_adam_acc_eps0.5 | 0.8539 |
| tuned_adam_acc_eps1.0 | 0.8531 |
| tuned_adam_acc_eps2.0 | 0.8541 |
| tuned_adam_accuracy_std | 0.0007722 |
| improvement | 0.005159 |
| improved_minus_tuned_adam_accuracy | 0.0001365 |
| majority_class_accuracy | 0.7607 |
| nonprivate_logreg_accuracy | 0.8533 |

Run on 2026-10-18. Full output: [results.json](results.json) · [run.log](run.log)

Notes: Split: 20% test held out once, remaining 80% split into 60% train / 20% validation (of total); tuning used validation only, test used once at the end. Hyperparameter-search privacy cost is not accounted for. Tuning used 2 separate noise seeds (100,101); final runs use seeds 42-46 (std is across seeds of the eps-averaged accuracy). Accountant is a simple RDP bound with integer orders. Tuned DP-Adam is a reference arm. Chosen hyperparameters: {"baseline_eps0.5": ["adam", 0.01, 1.0], "improved_eps0.5": ["sgdm", 3.0, 0.3], "tuned_adam_eps0.5": ["adam", 0.03, 1.0], "baseline_eps1.0": ["adam", 0.01, 1.0], "improved_eps1.0": ["sgdm", 3.0, 0.3], "tuned_adam_eps1.0": ["adam", 0.1, 0.3], "baseline_eps2.0": ["adam", 0.01, 1.0], "improved_eps2.0": ["sgdm", 1.0, 1.0], "tuned_adam_eps2.0": ["adam", 0.1,

## Reference

This project is motivated by the following paper:

> E Monzio Compagnoni, A Stanghellini (2026). *Adaptive Methods Are Preferable in High Privacy Settings: An SDE Perspective*. International. https://proceedings.iclr.cc/paper_files/paper/2026/hash/8cac7857c9412423eb11c16847c827f4-Abstract-Conference.html

Paper author: Prof. Aurelien Lucchi, University of Basel (aurelien.lucchi@unibas.ch)


---
_The code in this repository was drafted with AI assistance and runs automatically on GitHub Actions; results come from the latest run._
