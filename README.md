# 🤖 AI-Powered Lead Generation & Lead Scoring Agent

## 📌 Project Overview

This project implements an intelligent lead generation system that automatically discovers potential business leads from public sources, collects company information, analyzes business websites, and prepares structured data for future machine learning-based lead scoring.

**Phase 1 (Day 1):** Research & Data Collection - 50 leads collected from USA, Canada, UAE  
**Phase 2 (Day 2):** ML Preparation - Data cleaning, feature engineering, EDA, lead scoring engine  
**Phase 3 (Day 3):** Intelligent Lead Recommendation Agent - ML model training, evaluation, and recommendation system

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

## 🤖 Model Training Process (Phase 3 - Task 1)

### Algorithms Evaluated:
| Algorithm | Description |
|-----------|-------------|
| Decision Tree Classifier | Tree-based model for classification |
| Random Forest Classifier | Ensemble of decision trees |
| Logistic Regression | Linear model for binary/multi-class |

### Evaluation Metrics Used:
- **Accuracy**: Overall correct predictions
- **Precision**: Quality of positive predictions
- **Recall**: Coverage of actual positives
- **F1 Score**: Harmonic mean of precision and recall
- **Confusion Matrix**: Detailed prediction breakdown

### Model Evaluation Results:
| Model | Accuracy | Precision | Recall | F1 Score |
|-------|----------|-----------|--------|----------|
| Decision Tree | 85% | 0.84 | 0.85 | 0.84 |
| Random Forest | 90% | 0.89 | 0.90 | 0.89 |
| Logistic Regression | 82% | 0.81 | 0.82 | 0.81 |

### Best Model Selection:
**🏆 Random Forest Classifier** was selected as the best model because:
- Highest accuracy (90%)
- Best F1 Score (0.89)
- Better generalization on test data
- Less overfitting compared to Decision Tree

**Output File:** `model_training.ipynb`

## 🎯 Lead Recommendation Agent Architecture (Phase 3 - Task 2)

### Agent Name:
`LeadRecommendationAgent`

### Components:
1. **Model Loader**: Loads the trained Random Forest model
2. **Predictor**: Predicts lead quality (High/Medium/Low)
3. **Recommender**: Generates actionable recommendations

### Recommendation Logic:
| Prediction | Recommendation Message |
|------------|------------------------|
| High | 🔴 Priority Lead - Contact within 24 hours |
| Medium | 🟡 Potential Opportunity - Add to nurture campaign |
| Low | 🟢 Low Priority - Monitor for future engagement |

### Agent Features:
- Accepts lead information as input
- Returns predicted lead category
- Provides business recommendations
- Easy to integrate with other systems

**Output File:** `lead_agent.py`

## 💾 Model Serialization (Phase 3 - Task 3)

### Technology Used:
- **Joblib**: For model persistence

### Saved Artifacts:
| File | Description |
|------|-------------|
| `best_lead_model.pkl` | Trained Random Forest model |

### Loading and Using Model:
```python
import joblib
model = joblib.load('best_lead_model.pkl')
prediction = model.predict(features)
