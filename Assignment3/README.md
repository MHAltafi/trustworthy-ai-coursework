# Assignment 3 — Trustworthy AI

This assignment covers **Structural Causal Models (SCMs), causal inference, interventions, and confounding** through the implementation of an SCM class and several causal inference case studies.

## Structural Causal Model (SCM)

A general-purpose `SCM` class is implemented from scratch using **Python**, **Pandas**, and **NetworkX**.

The class supports:

* **Adding variables:** Variables, their parent variables, and structural equations can be added to the model.
* **Sampling:** Random samples are generated in topological order from the joint distribution.
* **Interventions:** The `do_operator` applies interventions by replacing the structural equation of an intervened variable and removing its incoming causal edges.
* **Graph visualization:** The corresponding causal DAG can be plotted using NetworkX and Matplotlib.

## Causal Effect of an Educational Course

A causal model is constructed to study the effect of an intensive university entrance exam preparation course.

The model contains:

* `U`: Unobserved motivation
* `X`: Course enrollment
* `Y`: Exam score

The hidden motivation variable acts as a **confounder**, affecting both course enrollment and exam performance.

Using observational and interventional samples, the statistical association between course enrollment and exam score is compared with the true causal effect obtained through interventions on `X`.

## Causal Effect of Smoking on Heart Disease

A second SCM is constructed to model the relationships between demographic, socioeconomic, genetic, lifestyle, and biological factors affecting heart disease risk.

The model includes:

* Age
* Socioeconomic status (SES)
* Genetic risk
* Smoking
* BMI
* Blood pressure
* Cholesterol
* Heart disease risk

The causal graph is constructed and visualized using the implemented SCM class.

Interventions `do(Smoking = 1)` and `do(Smoking = 0)` are used to estimate:

* **ATE:** Average Treatment Effect
* **ATT:** Average Treatment Effect on the Treated
* **CATE:** Conditional Average Treatment Effect for individuals older than 55 and those aged 55 or younger

The resulting causal effects are analyzed and compared across the population and age groups.

## Website Layout and Simpson's Paradox

The final case study analyzes `website_engagement_data.csv`, where users were assigned to old or new website layouts according to their age group.

The analysis is divided into four stages.

### Naive Analysis

The average time spent on the old and new layouts is compared, followed by a simple OLS regression:

`time_spent ~ layout_code`

The resulting association is evaluated for statistical significance and used to illustrate how a naive analysis can lead to a misleading product recommendation.

### Visual Diagnosis

The distribution of users across layouts and age groups is visualized to reveal the imbalance in layout assignment.

Boxplots are also used to compare time spent within each age group, illustrating **Simpson's paradox** and showing why the aggregated comparison can be misleading.

### Controlled Causal Analysis

A causal DAG is constructed in which **age group acts as a confounder** affecting both layout assignment and time spent.

A controlled OLS regression is then fitted:

`time_spent ~ layout_code + age_code`

The controlled estimate is compared with the naive estimate to investigate the effect of confounding and the resulting change in the estimated effect of the new layout.

### Placebo Test

As a robustness check, the layout assignment is randomly shuffled while keeping the age group unchanged. The controlled regression is then repeated using the shuffled layout variable.

The placebo result is examined to determine whether the observed layout effect can be reproduced when the treatment assignment is randomized.

## Notebook

The complete implementation, causal graphs, simulations, statistical analyses, visualizations, and results are available in:

`notebooks/Assignment3.ipynb`
