# Assignment 4 — Trustworthy AI

This assignment focuses on **fairness, bias, privacy, and trustworthy machine learning** through two complementary problems. The first part studies fairness in automated decision-making using the COMPAS recidivism dataset, while the second investigates membership privacy risks and differential privacy in federated learning.

## Assignment 4-1 — COMPAS Fairness Audit

This part investigates **algorithmic fairness and bias** in the COMPAS recidivism prediction system.

### Dataset and Bias Audit

The ProPublica COMPAS dataset is used to predict two-year recidivism, with **race** treated as the sensitive attribute.

The initial audit examines:

* Recidivism rates across racial groups.
* Distribution of COMPAS decile scores by race.
* The ability to predict race from non-race features.
* Potential proxy features that may indirectly encode race.

A logistic regression model is then trained as the baseline predictive model.

### Fairness Evaluation

Several group fairness metrics are implemented from scratch, including:

* Demographic parity / selection rate.
* True Positive Rate (TPR) and Equal Opportunity.
* False Positive Rate (FPR).
* Positive Predictive Value (PPV).
* Base rates and corresponding between-group gaps.

The baseline model is compared with a **Fairness Through Unawareness** version in which the race feature is removed from the model inputs. This allows the effect of indirect proxies for race to be investigated.

### Fairness Interventions

Two different fairness interventions are implemented.

**Preprocessing:** Calders–Kamiran reweighting is applied to modify the importance of training examples according to the joint distribution of race and the target.

**Postprocessing:** Group-specific decision thresholds are selected using a validation set to reduce differences in TPR and improve Equal Opportunity.

The resulting models are compared in terms of predictive performance and fairness.

### Fairness–Accuracy Trade-off

Decision thresholds are swept to study the trade-off between predictive accuracy and different fairness criteria.

The analysis also investigates the empirical incompatibility between:

* Demographic parity.
* Equal opportunity.
* Equal PPV.

The relationship between base rates, TPR, FPR, FNR, and PPV is examined to understand why these fairness criteria cannot generally be satisfied simultaneously when groups have different base rates.

### Intersectional Fairness

The final audit considers **race × sex intersectional groups** rather than race alone.

Group-specific base rates, selection rates, TPR, FPR, PPV, and accuracy are examined to determine whether marginal fairness can hide disparities within smaller intersectional subgroups.

### Notebook

The complete implementation is available in:

`notebooks/Assignment4-1.ipynb`

## Assignment 4-2 — Federated Learning and Subject Membership Inference

This part investigates **privacy risks in federated learning** and the use of differential privacy to mitigate membership leakage.

### Dataset and Federated Data Partitioning

The **FEMNIST** dataset is analyzed at the subject level, including the number of images, classes, subjects, and images per subject.

Subjects are divided into two disjoint groups:

* **Member Subjects:** their data is allowed to participate in federated training.
* **Non-Member Subjects:** their data is completely excluded from federated training.

The member data is distributed across **8 federated silos**. A subject may appear in one or multiple organizations, allowing the effect of cross-organizational subject distribution to be studied.

The silo sizes and subject distribution are analyzed to assess the balance of the federated system and its potential effect on membership inference.

### Federated Learning

A convolutional neural network is trained using **Federated Averaging (FedAvg)**.

Each communication round consists of:

1. Sending the global model to the participating silos.
2. Training locally on each silo.
3. Sending the updated model parameters to the server.
4. Aggregating the local models using data-size-weighted averaging.

Only Member Subjects participate in federated training, ensuring a clear distinction between members and non-members for the subsequent privacy attacks.

Training and test performance are monitored across federated rounds.

### Subject-Level Membership Inference

The main attack is a **Subject-Level Loss-Across-Rounds Membership Inference Attack**.

For each subject, the average loss of its records is collected from the global model at each federated round. These loss values form a trajectory that captures how the model's behavior changes throughout training.

Member and non-member trajectories are compared to construct a membership score.

The attack is evaluated using:

* Accuracy
* Precision
* Recall
* F1-score
* ROC-AUC

Attack performance is also tracked across federated rounds to study how membership information accumulates during training.

### Record-Level Membership Inference

A baseline **Record-Level Membership Inference Attack** is implemented using individual record losses.

The record-level and subject-level attacks are compared to investigate whether membership can be inferred more effectively at the individual-record or subject level, and what this implies for privacy in federated learning.

### Differential Privacy

Two privacy defenses are investigated.

**Record-Level DP:** DP-SGD is applied using per-record gradient clipping and Gaussian noise, with privacy accounting based on the target privacy budget.

**Subject-Level DP:** records belonging to the same subject are grouped together, and subject-level gradients are clipped before noise is added. This changes the privacy unit from an individual record to an entire subject.

Both defenses are evaluated using model accuracy and the performance of the membership inference attacks.

### Privacy Sensitivity and Trade-offs

The effect of changing the main DP parameters is investigated by:

* Increasing the noise multiplier.
* Reducing the clipping norm.
* Increasing the number of federated rounds.

The final comparison considers privacy budget, noise level, clipping norm, model accuracy, and membership inference performance.

The results are used to study the **privacy–utility trade-off** and to examine why subject-level differential privacy can be particularly relevant when the privacy unit is an individual subject whose data may be distributed across multiple organizations.

### Notebook

The complete implementation is available in:

`notebooks/Assignment4-2.ipynb`
