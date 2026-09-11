# Trustworthy AI — University of Tehran

This repository contains my coursework and implementations for **Trustworthy AI** course at **University of Tehran (Semester 4042)**.

The assignments cover several important aspects of trustworthy machine learning, including **spurious correlations, adversarial robustness, interpretability, causal inference, fairness, privacy, and federated learning**. The implementations are primarily developed in Python using libraries such as PyTorch, Scikit-learn, Pandas, NumPy, and NetworkX.

> **Note:** The first two assignments were completed during a period of widespread internet disruption in Iran.

## Assignments

### Assignment 1 — Spurious Correlation and Adversarial Robustness

The first assignment focuses on two aspects of model reliability.

**Assignment 1-1 — Spurious Correlation and JTT**

* Colored MNIST and distribution shifts.
* Empirical Risk Minimization (ERM).
* Identification of incorrectly classified samples.
* Just Train Twice (JTT) style upsampling.
* Generalized Cross-Entropy (GCE).

[**Notebook: Assignment1-1.ipynb**](Assignment1/notebooks/Assignment1-1.ipynb)

**Assignment 1-2 — Adversarial Attacks**

* White-box adversarial attacks on a pretrained ResNet-20.
* Fast Gradient Sign Method (FGSM).
* Projected Gradient Descent (PGD).
* Visualization and evaluation of adversarial examples.

[**Notebook: Assignment1-2.ipynb**](Assignment1/notebooks/Assignment1-2.ipynb)

### Assignment 2 — Model Interpretability and Explainability

The second assignment focuses on understanding and visualizing the decisions of deep learning models.

**Assignment 2-1 — LIME**

* Local interpretable explanations for image classification.
* Custom image segmentation using K-means.
* Generation of perturbed samples.
* LIME kernel weighting.
* Weighted Lasso regression.
* Visualization of segment-level feature importance.

[**Notebook: Assignment2-1.ipynb**](Assignment2/notebooks/Assignment2-1.ipynb)

**Assignment 2-2 — Grad-CAM and Finer-CAM**

* Grad-CAM implemented from scratch.
* Guided Backpropagation and Guided Grad-CAM.
* Finer-CAM implementation.
* Comparison of different explanation maps.
* Analysis of competing class logits and the effect of the `γ` parameter.

[**Notebook: Assignment2-2.ipynb**](Assignment2/notebooks/Assignment2-2.ipynb)

### Assignment 3 — Causal Inference

The third assignment focuses on **Structural Causal Models (SCMs), interventions, confounding, and causal effects**.

The implementation includes:

* A general-purpose SCM class built from scratch.
* Topological sampling and causal graph visualization.
* The `do`-operator for interventions.
* Comparison of observational and interventional effects.
* Causal analysis of an educational course.
* A causal model for smoking and heart disease.
* Estimation of ATE, ATT, and CATE.
* Analysis of confounding and Simpson's paradox.
* Causal analysis of website layout experiments.
* Controlled regression and placebo testing.

[**Notebook: Assignment3.ipynb**](Assignment3/notebook/Assignment3.ipynb)

### Assignment 4 — Fairness and Privacy

The fourth assignment focuses on **fairness, algorithmic bias, privacy, and federated learning**.

**Assignment 4-1 — COMPAS Fairness Audit**

* Bias analysis of the COMPAS recidivism dataset.
* Fairness evaluation across racial groups.
* Demographic parity, TPR, FPR, PPV, and related fairness gaps.
* Fairness Through Unawareness.
* Calders–Kamiran preprocessing.
* Equal Opportunity postprocessing.
* Fairness–accuracy trade-offs.
* Fairness impossibility analysis.
* Race × sex intersectional fairness analysis.

[**Notebook: Assignment4-1.ipynb**](Assignment4/notebooks/Assignment4-1.ipynb)

**Assignment 4-2 — Federated Learning and Subject Membership Inference**

* Subject-level Member / Non-Member splitting on FEMNIST.
* Cross-silo federated learning with 8 organizations.
* Federated Averaging (FedAvg).
* Subject-level membership inference using loss trajectories.
* Record-level membership inference.
* Record-level Differential Privacy with DP-SGD.
* Subject-level Differential Privacy.
* Privacy sensitivity analysis.
* Privacy–utility trade-offs between different defenses.

[**Notebook: Assignment4-2.ipynb**](Assignment4/notebooks/Assignment4-2.ipynb)

## Repository Structure

```text
trustworthy-ai-coursework/
│
├── Assignment1/
│   ├── notebooks/
│   │   ├── Assignment1-1.ipynb
│   │   └── Assignment1-2.ipynb
│   ├── models/
│   │   └── cifar10_resnet20-4118986f.pt
│   └── README.md
│
├── Assignment2/
│   ├── notebooks/
│   │   ├── Assignment2-1.ipynb
│   │   └── Assignment2-2.ipynb
│   └── README.md
│
├── Assignment3/
│   ├── notebook/
│   │   └── Assignment3.ipynb
│   ├── data/
│   └── README.md
│
├── Assignment4/
│   ├── notebooks/
│   │   ├── Assignment4-1.ipynb
│   │   └── Assignment4-2.ipynb
│   └── README.md
│
├── .gitignore
└── README.md
```

## Technologies

The coursework uses a range of Python tools and machine learning frameworks, including:

* **Python**
* **PyTorch**
* **Torchvision**
* **Scikit-learn**
* **Pandas**
* **NumPy**
* **Matplotlib**
* **NetworkX**
* **Opacus**

Different assignments use different subsets of these libraries depending on the task.

## Topics Covered

Across the four assignments, the main trustworthy AI topics covered in this repository are:

| Topic                             | Assignment   |
| --------------------------------- | ------------ |
| Spurious Correlations             | Assignment 1 |
| JTT and Robust Training           | Assignment 1 |
| Adversarial Robustness            | Assignment 1 |
| Model Interpretability            | Assignment 2 |
| LIME                              | Assignment 2 |
| Grad-CAM / Finer-CAM              | Assignment 2 |
| Structural Causal Models          | Assignment 3 |
| Causal Inference                  | Assignment 3 |
| Confounding and Simpson's Paradox | Assignment 3 |
| Algorithmic Fairness              | Assignment 4 |
| Fairness Interventions            | Assignment 4 |
| Federated Learning                | Assignment 4 |
| Membership Inference              | Assignment 4 |
| Differential Privacy              | Assignment 4 |

## Reproducibility

Each assignment contains its own notebooks with the complete implementation, experiments, visualizations, and analysis.

Where applicable, random seeds and experimental parameters are explicitly defined in the notebooks to make the experiments reproducible.

Detailed methodology and assignment-specific information can be found in the `README.md` files inside each assignment directory.

