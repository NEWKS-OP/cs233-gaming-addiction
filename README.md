# Predicting Gaming Addiction — CS-233 course project

Group project for **CS-233 Introduction to Machine Learning** at EPFL, spring
2026. Team: SCIPER 392443 (Saad Amraoui), 400305 and 399383.

**Data.** The "Gaming and Mental Health" dataset: 2,000 samples with 13
numerical features about gaming habits, split into 1,600 for training and 400
for testing.

**Tasks.**
- **Regression:** predict the addiction score, an integer in [0, 10].
- **Classification:** predict the addiction level, one of three classes (Low,
  Medium, High).

All methods are implemented from scratch in NumPy.

| | Methods | Report |
|---|---|---|
| [Milestone 1](milestone1) | Linear regression, logistic regression, k-NN | [`milestone1/report.pdf`](milestone1/report.pdf) |
| [Milestone 2](milestone2) | MLP (backpropagation, mini-batch SGD), K-Means | [`milestone2/report.pdf`](milestone2/report.pdf) |

## Shared protocol

- **Normalisation.** Features are standardised with a z-score. The mean and
  standard deviation are computed on the training set only.
- **Model selection.** Hyper-parameters are chosen on a validation set made of
  20 % of the training data.
- **Test set.** It is used once, for the final numbers.

## Milestone 1: linear models and k-NN

| Method | Task | Test accuracy / macro-F1 | Test MSE |
|---|---|---|---|
| Linear regression (normal equation) | Regression | – | 0.994 |
| Logistic regression (softmax, GD, η = 1e-3, 1500 iters) | Classification | 88.75 % / 0.863 | – |
| k-NN (k = 15) | Classification | 80.00 % / 0.519 | – |
| k-NN (k = 12) | Regression | – | 1.858 |

## Milestone 2: neural network and clustering

| Method | Task | Test accuracy / macro-F1 | Test MSE |
|---|---|---|---|
| MLP [64, 32], η = 0.01 | Classification | 85.2 % / 0.755 | – |
| MLP [8, 8], η = 3e-3 | Regression | – | 1.03 |
| K-Means, K* = 35 (elbow), majority vote | Classification | 73.8 % / 0.635 | – |

The regression MLP improves on milestone 1's k-NN baseline (test MSE 1.86).
For K-Means, the number of clusters chosen without labels, from the
inertia/silhouette elbow, gives almost the same F1 as the best K chosen with
labels (0.534 vs. 0.578 on validation).

## Run

The dataset (`data/features.npz`) was provided by the course and is not
included in this repo. Put it in `milestone1/data/` or `milestone2/data/`, then:

```bash
cd milestone1
python main.py --task classification --method logistic_regression --lr 1e-3 --max_iters 1500 --test

cd ../milestone2
python main.py --task classification --method kmeans --K 35 --test
python main.py --task classification --method mlp --lr 1e-2 --test
```

## Credits

The skeleton code was provided by the CS-233 teaching staff:
- `src/utils.py`, `dummy_methods.py`;
- the test scripts `test_ms1.py` and `test_ms2.py`;
- the structure of `main.py`.
