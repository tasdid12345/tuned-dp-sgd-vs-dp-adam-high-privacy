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

## Reference

This project is motivated by the following paper:

> E Monzio Compagnoni, A Stanghellini (2026). *Adaptive Methods Are Preferable in High Privacy Settings: An SDE Perspective*. International. https://proceedings.iclr.cc/paper_files/paper/2026/hash/8cac7857c9412423eb11c16847c827f4-Abstract-Conference.html

Paper author: Prof. Aurelien Lucchi, University of Basel (aurelien.lucchi@unibas.ch)


---
_The code in this repository was drafted with AI assistance and runs automatically on GitHub Actions; results come from the latest run._
