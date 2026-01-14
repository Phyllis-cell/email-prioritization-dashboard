#  Email Prioritization System (Model + Dashboard)

This repository contains an end-to-end **email prioritization system** that classifies incoming emails into actionable priority levels and includes an interactive dashboard for testing and evaluation.

The project demonstrates **data preprocessing, rule-guided labeling, machine learning modeling, explainability, evaluation metrics, and deployment readiness**.

---

##  Problem Statement

Design a system that processes simulated email events using:
- **Sender (From)**
- **Subject**
- **Body**

and assigns one of the following priority labels:

- **Prioritize** – critical or security-related emails (MFA, verification, password reset)
- **Default** – general informational emails
- **Slow** – promotional or non-urgent emails

Each prediction must include:
- Priority label
- Human-readable reasoning
- Confidence score

---

##  Solution Overview

### 1. Email Parsing
Raw email text is parsed into:
- `from`
- `subject`
- `body`

This ensures the model explicitly consumes realistic email signals rather than raw text only.

---

### 2. Priority Label Generation
Rule-based logic is used to generate training labels:
- Security keywords → **Prioritize**
- Marketing keywords → **Slow**
- Otherwise → **Default**

These labels simulate realistic email prioritization behavior.

---

### 3. Text Preprocessing
A lightweight preprocessing pipeline is applied:
- Email header removal
- URL / EMAIL / NUM token preservation
- Text normalization (lowercasing, spacing)

This avoids over-cleaning while preserving intent.

---

### 4. Model
- **TF-IDF Vectorization** (unigrams + bigrams)
- **Logistic Regression**

Chosen for:
- Simplicity
- Interpretability
- Strong baseline performance for text classification

---

### 5. Explainability
Each prediction returns:
- **Label** – predicted priority
- **Reasoning** – pattern-based explanation
- **Confidence** – model probability score

This ensures the system is not a black box.

---

##  Evaluation

Model evaluation includes:
- Accuracy
- Macro-F1 score
- Confusion matrix
- Classification report

These metrics are displayed both in code and in the dashboard UI.

---

##  Interactive Dashboard

The Gradio dashboard allows users to:
- Submit test email events (sender, subject, body)
- View predictions in real time
- Inspect model performance metrics

---

## 🛠 Tech Stack

- Python
- scikit-learn
- pandas, numpy
- matplotlib
- Gradio

---

##  Repository Structure

```
.
├── app.py
├── requirements.txt
├── email_classification_dataset.csv
└── README.md
```

---

##  Run Locally

```bash
pip install -r requirements.txt
python app.py
```

---

##  Deployment Note

The application is **deployment-ready** and can be hosted using:
- Hugging Face Spaces (Gradio)
- Any standard Python hosting environment

Deployment is optional depending on evaluation requirements.

---
# LINKS
Live App: https://huggingface.co/spaces/Phyllis10/email-prioritization-dashboard
Source Code: (your GitHub repo link)
Video Demo: (Loom link — “coming soon” for now)


##  Author
**Phyllis Barikisu Snyper**  
Data Science Engineer & AI Lead
