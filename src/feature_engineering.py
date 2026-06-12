import pandas as pd

# Load cleaned data
df = pd.read_csv('data/cleaned_leads.csv')

print("Available columns:", list(df.columns))
print(f"Initial shape: {df.shape}")

# 1. website_exists (1 if website exists)
df['website_exists'] = df['Website'].apply(lambda x: 1 if pd.notna(x) else 0)

# 2. contact_form (1 if contact form exists)
df['contact_form'] = df['Contact Form'].apply(lambda x: 1 if x == 'Yes' else 0)

# 3. services_count (count services by comma)
df['services_count'] = df['Company Services'].apply(
    lambda x: len(str(x).split(',')) if pd.notna(x) else 0
)

# 4. contact_page_exists (1 if contact page exists)
df['contact_page_exists'] = df['Contact Page'].apply(
    lambda x: 1 if pd.notna(x) and str(x).strip() != '' else 0
)

# 5. country_score
country_scores = {
    'United States': 3,
    'Canada': 2,
    'United Arab Emirates': 1
}
df['country_score'] = df['Country'].map(country_scores)
# Fill NaN values with 0 (countries not in scoring list)
df['country_score'] = df['country_score'].fillna(0)

# 6. lead_quality_score
quality_scores = {
    'High': 3,
    'Medium': 2,
    'Low': 1
}
df['lead_quality_score'] = df['Lead Quality'].map(quality_scores)

# 7. industry_score (optional - assign scores based on industry relevance)
industry_scores = {
    'Technology': 3,
    'Software': 3,
    'Marketing': 2,
    'E-commerce': 2,
    'Retail': 1,
    'Manufacturing': 1
}
df['industry_score'] = df['Industry'].map(industry_scores)
df['industry_score'] = df['industry_score'].fillna(1)  # Default score for unknown industries

# 8. total_lead_score (composite score)
df['total_lead_score'] = (
    df['website_exists'] * 1 +
    df['contact_form'] * 2 +
    df['contact_page_exists'] * 1 +
    df['services_count'].clip(upper=5) +  # Cap at 5 services
    df['country_score'] +
    df['lead_quality_score'] * 2 +  # Give more weight to lead quality
    df['industry_score']
)

# Select only the columns we want for ML
ml_columns = [
    'Company Name',
    'Website', 
    'Country',
    'Industry',
    'website_exists',
    'contact_form',
    'contact_page_exists',
    'services_count',
    'country_score',
    'industry_score',
    'lead_quality_score',
    'total_lead_score'
]

df_ml = df[ml_columns]

# Save
df_ml.to_csv('data/ml_ready_dataset.csv', index=False)
print("\n✅ Feature engineering complete!")
print(f"📊 Final dataset shape: {df_ml.shape}")
print(f"\n📋 First few rows:")
print(df_ml.head())
print(f"\n📈 Total lead score statistics:")
print(df_ml['total_lead_score'].describe())