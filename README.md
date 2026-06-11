# 🤖 AI-Powered Lead Generation & Lead Scoring Agent

## 📌 Project Overview

This project implements an intelligent lead generation system that automatically discovers potential business leads from public sources, collects company information, analyzes business websites, and prepares structured data for future machine learning-based lead scoring.

**Phase 1 (Day 1):** Research & Data Collection - 50 leads collected from USA, Canada, UAE  
**Phase 2 (Day 2):** ML Preparation - Data cleaning, feature engineering, EDA, lead scoring engine

## 🎯 Business Problem

Sales and marketing teams spend approximately 40% of their time manually searching for potential leads and qualifying them. This manual process is time-consuming, inconsistent, not scalable, and leads to delayed responses.

## 📊 Dataset Cleaning Process (Task 1)

### Steps Performed:
1. Removed duplicate records
2. Removed rows with missing websites
3. Standardized country names (USA → United States, UAE → United Arab Emirates)
4. Standardized Contact Page values (Yes/No)
5. Fixed spelling inconsistencies in Lead Quality column

### Cleaning Results:
| Metric | Value |
|--------|-------|
| Initial Records | 50 |
| Duplicates Removed | 0 |
| Missing Website Removed | 0 |
| Final Cleaned Records | 50 |

**Output File:** `cleaned_leads.csv`

## 🔧 Feature Engineering Approach (Task 2)

### New Features Created:
| Feature | Description | Values |
|---------|-------------|--------|
| website_exists | Website available | 1=Yes, 0=No |
| contact_form | Contact form available | 1=Yes, 0=No |
| services_count | Number of services | Integer |
| about_word_count | About section words | Integer |
| country_score | Country score | US=3, Canada=2, UAE=1 |
| lead_quality_score | Quality score | High=3, Medium=2, Low=1 |

**Output File:** `ml_ready_dataset.csv`

## 📈 Exploratory Data Analysis Summary (Task 3)

### Key Findings:
| Analysis | Result |
|----------|--------|
| Total Companies | 50 |
| Companies by Country | USA: 17, Canada: 17, UAE: 16 |
| Lead Quality Distribution | High: 20 (40%), Medium: 15 (30%), Low: 15 (30%) |
| Average Services per Company | 2.8 |
| Companies with Contact Forms | 35 (70%) |

### Visualizations Created:
- Country Distribution Chart
- Industry Distribution Chart
- Lead Quality Chart

**Output File:** `lead_eda.ipynb`

## 🎯 Lead Scoring Logic (Task 4)

### Scoring Rules:
| Condition | Score |
|-----------|-------|
| Website Available | +10 |
| Contact Form Available | +20 |
| United States Company | +15 |
| Canada Company | +10 |
| United Arab Emirates Company | +5 |
| Services Count > 3 | +15 |
| About Section > 100 words | +10 |

### Classification:
| Score Range | Class |
|-------------|-------|
| 60 and above | High |
| 35 - 59 | Medium |
| Below 35 | Low |

### Scoring Results:
| Class | Count | Percentage |
|-------|-------|------------|
| High | 22 | 44% |
| Medium | 18 | 36% |
| Low | 10 | 20% |

**Output File:** `scored_leads.csv`

## 🧠 ML Dataset Preparation (Task 5)

### Features Selected (X):
- website_exists
- contact_form
- services_count
- about_word_count
- country_score

### Target Variable (y):
- lead_quality_score

### Train-Test Split (80:20):
| Dataset | Samples |
|---------|---------|
| Training Set | 40 |
| Testing Set | 10 |

**Output File:** `ml_preparation.ipynb`

## 🔮 Future Model Training Plan

### Phase 3: Actual ML Model Building

**Algorithms to Test:**
- Logistic Regression
- Random Forest Classifier
- XGBoost

**Evaluation Metrics:**
- Accuracy, Precision, Recall, F1-Score

**Expected Output:**
- Trained model for lead scoring


## 📚 Key Learnings

### From Day 1 (Data Collection):
- Lead generation from public sources
- Website analysis techniques
- Lead qualification criteria

### From Day 2 (ML Preparation):
- Data cleaning and standardization
- Feature engineering for ML
- Exploratory Data Analysis
- Rule-based scoring systems
- Train-test split preparation

## 📦 Submission Files

| File | Description |
|------|-------------|
| cleaned_leads.csv | Cleaned dataset |
| ml_ready_dataset.csv | Feature-engineered dataset |
| scored_leads.csv | Leads with scores |
| lead_eda.ipynb | EDA with charts |
| ml_preparation.ipynb | Train-test split |
| README.md | Documentation |

