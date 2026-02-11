# 🔍 Explainable Anomaly Detection for Financial Audits

An **explainable machine learning system** that detects anomalous financial transactions and provides **human-understandable explanations** for each flagged transaction — built for transparency in financial audits.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-orange?logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-Data-green?logo=pandas&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## 📋 Table of Contents

- [Introduction](#-introduction)
- [Problem Statement](#-problem-statement)
- [Features](#-features)
- [Project Architecture](#-project-architecture)
- [Tech Stack](#-tech-stack)
- [Getting Started](#-getting-started)
- [Step-by-Step Beginner Guide](#-step-by-step-beginner-guide-to-building-this-project)
  - [Step 1: Set Up Your Environment](#step-1-set-up-your-environment)
  - [Step 2: Generate or Load the Dataset](#step-2-generate-or-load-the-dataset)
  - [Step 3: Explore the Data (EDA)](#step-3-explore-the-data-eda)
  - [Step 4: Preprocess the Data](#step-4-preprocess-the-data)
  - [Step 5: Train the Anomaly Detection Model](#step-5-train-the-anomaly-detection-model)
  - [Step 6: Generate Explanations](#step-6-generate-explanations-for-anomalies)
  - [Step 7: Visualize the Results](#step-7-visualize-the-results)
  - [Step 8: Build a Summary Report](#step-8-build-a-summary-report)
- [Project Structure](#-project-structure)
- [Sample Output](#-sample-output)
- [Future Enhancements](#-future-enhancements)
- [Contributing](#-contributing)
- [License](#-license)

---

## 📖 Introduction

Financial institutions generate massive volumes of transaction data daily, making manual auditing nearly impossible. While automated anomaly detection can help, most existing systems act as **black boxes** — they flag transactions without explaining *why*.

This project solves that problem by building a system that:
1. **Detects anomalous transactions** using the Isolation Forest algorithm
2. **Explains each anomaly** by analyzing feature-level deviations in plain language
3. **Visualizes results** with clear, audit-friendly charts and reports

---

## 🎯 Problem Statement

> Design a machine learning–based system that detects anomalous financial transactions from historical data and provides simple, interpretable explanations for each detected anomaly.

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 🔎 **Anomaly Detection** | Uses Isolation Forest to identify unusual transactions |
| 📝 **Explainable Results** | Generates plain-English explanations for every flagged transaction |
| 📊 **Visual Dashboards** | Distribution plots, scatter plots, and anomaly highlighting |
| 📁 **Structured Reports** | Exports anomaly reports to CSV for auditors |
| 🧪 **Synthetic Data Generator** | Includes a script to generate realistic financial transaction data |

---

## 🏗 Project Architecture

The system follows a pipeline architecture with the following stages:

1. **Transaction Data** (CSV / Synthetic)
2. **Data Preprocessing** - Missing values, Normalization, Encoding
3. **Isolation Forest** (Anomaly Detection)
4. **Explainability Engine** (Feature Deviation Analysis)
5. **Visualizations & Audit Report**

---

## 🛠 Tech Stack

| Tool | Purpose |
|------|---------|
| **Python 3.8+** | Core programming language |
| **Pandas** | Data manipulation and analysis |
| **NumPy** | Numerical computations |
| **Scikit-learn** | Isolation Forest model |
| **Matplotlib** | Static visualizations |
| **Seaborn** | Statistical visualizations |

---

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher installed on your system
- `pip` package manager
- A code editor (VS Code, PyCharm, or Jupyter Notebook)

### Installation

Follow these steps to get started:
1. Clone the repository
2. Create a virtual environment (recommended)
3. Install dependencies using pip

---

## 📚 Step-by-Step Beginner Guide to Building This Project

> This guide walks you through building the entire project from scratch. Each step includes **what** you're doing, **why** you're doing it, and the **code** to do it.

---

### Step 1: Set Up Your Environment

**What:** Install Python and the required libraries.

**Why:** These libraries provide the tools for data handling, machine learning, and visualization.

Create a new Python file (e.g., `main.py`) or a Jupyter Notebook to follow along.

---

### Step 2: Generate or Load the Dataset

**What:** Create a synthetic financial transaction dataset (or load your own CSV).

**Why:** We need transaction data with features like amount, frequency, and category to train our model. Synthetic data lets us get started without needing real financial data.

**Key columns explained:**
| Column | Meaning |
|--------|---------|
| `amount` | Transaction dollar amount |
| `transaction_frequency` | How many transactions the user made recently |
| `avg_balance` | Average account balance |
| `category` | Type of purchase |
| `is_weekend` | Whether the transaction happened on a weekend |

---

### Step 3: Explore the Data (EDA)

**What:** Visualize and understand the data distribution before modeling.

**Why:** EDA helps you spot patterns, understand feature ranges, and verify that the data looks reasonable.

**💡 What to look for:**
- Are there clear outliers in the amount distribution?
- Do some users have unusually high transaction frequency?
- Is there a cluster of low-balance, high-amount transactions?

---

### Step 4: Preprocess the Data

**What:** Clean and transform the data so the ML model can use it.

**Why:** Machine learning models require numerical input. We need to encode categories and normalize feature scales.

**Key preprocessing steps explained:**
1. **Missing values** → Filled with median (robust to outliers)
2. **Label Encoding** → Converts categories like "groceries" to numbers like 0, 1, 2...
3. **StandardScaler** → Scales all features to have mean=0 and std=1 (important for Isolation Forest)

---

### Step 5: Train the Anomaly Detection Model

**What:** Use the **Isolation Forest** algorithm to find anomalous transactions.

**Why:** Isolation Forest works by randomly partitioning data — anomalies are easier to isolate (fewer partitions needed), so they get lower scores. It's unsupervised, meaning we don't need labeled "fraud" data.

**Understanding the output:**
- `anomaly_label = -1` → The model thinks this transaction is **anomalous**
- `anomaly_label = 1` → The model thinks this transaction is **normal**
- `anomaly_score` → A continuous score; **more negative = more anomalous**

---

### Step 6: Generate Explanations for Anomalies

**What:** For each flagged transaction, explain *which features* made it anomalous and *by how much*.

**Why:** This is the **core value** of the project. Auditors need to know *why* a transaction was flagged, not just *that* it was flagged.

---

### Step 7: Visualize the Results

**What:** Create clear, audit-friendly visualizations.

**Why:** Visual evidence makes it easier for auditors to understand and act on the flagged transactions.

---

### Step 8: Build a Summary Report

**What:** Export the results to a clean CSV report that auditors can review.

**Why:** This is the deliverable — a structured file listing each anomaly with its explanation.

---

## 📁 Project Structure

The project is organized with the following files:
- README.md - This file
- LICENSE - MIT License
- main.py - Complete pipeline script
- transactions.csv - Generated/input dataset
- audit_report.csv - Output anomaly report
- eda_distributions.png - EDA visualization
- anomaly_results.png - Results dashboard
- requirements.txt - Python dependencies

---

## 📊 Sample Output

### Detection Summary
The system outputs summary statistics showing total transactions, normal transactions, and anomalies detected.

### Example Explanation
Each anomaly includes detailed explanations showing:
- Transaction ID
- Anomaly Score
- Specific reasons why the transaction was flagged (e.g., amount, frequency, balance deviations)

---

## 🔮 Future Enhancements

- [ ] Add SHAP (SHapley Additive exPlanations) for model-level explainability
- [ ] Build a Streamlit / Flask web dashboard for interactive exploration
- [ ] Support real-world datasets (e.g., Kaggle credit card fraud dataset)
- [ ] Add more anomaly detection models (LOF, DBSCAN, Autoencoders)
- [ ] Implement time-series anomaly detection for temporal patterns
- [ ] Add email/Slack alerting for newly detected anomalies
- [ ] Dockerize the application for easy deployment

---

## 🤝 Contributing

Contributions are welcome! Here's how to get started:

1. **Fork** the repository
2. **Create** a feature branch
3. **Commit** your changes
4. **Push** to the branch
5. **Open** a Pull Request

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

<p align="center">
  Made with ❤️ for transparent financial auditing
</p>