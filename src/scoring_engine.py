import pandas as pd

# Load ML ready dataset
df = pd.read_csv('data/ml_ready_dataset.csv')

# Display available columns
print("Available columns for scoring:")
print(list(df.columns))
print("\n" + "="*50 + "\n")

# Calculate score for each company
def calculate_score(row):
    score = 0
    
    # 1. Website available (max 10 points)
    if row['website_exists'] == 1:
        score += 10
    
    # 2. Contact form available (max 20 points)
    if row['contact_form'] == 1:
        score += 20
    
    # 3. Contact page available (max 10 points) - New
    if 'contact_page_exists' in df.columns and row['contact_page_exists'] == 1:
        score += 10
    
    # 4. Country score (max 15 points)
    if row['Country'] == 'United States':
        score += 15
    elif row['Country'] == 'Canada':
        score += 10
    elif row['Country'] == 'United Arab Emirates':
        score += 5
    else:
        score += 2  # Small points for other countries
    
    # 5. Services count (max 15 points)
    if row['services_count'] > 5:
        score += 15
    elif row['services_count'] > 3:
        score += 10
    elif row['services_count'] > 1:
        score += 5
    elif row['services_count'] == 1:
        score += 2
    
    # 6. Lead quality score from data (max 9 points - already 3,2,1 multiplied by 3)
    if 'lead_quality_score' in df.columns:
        score += row['lead_quality_score'] * 3  # High=9, Medium=6, Low=3
    
    # 7. Industry score (max 5 points)
    if 'industry_score' in df.columns:
        score += row['industry_score']
    
    return score

# Apply scoring
df['Lead Score'] = df.apply(calculate_score, axis=1)

# Ensure score is within 0-100 range
df['Lead Score'] = df['Lead Score'].clip(0, 100)

# Classify based on score
def classify_score(score):
    if score >= 70:
        return 'High'
    elif score >= 40:
        return 'Medium'
    else:
        return 'Low'

df['Predicted Lead Class'] = df['Lead Score'].apply(classify_score)

# Display scoring results
print("📊 Scoring Results:")
print("="*50)
print(f"\nScore Statistics:")
print(f"  Mean Score: {df['Lead Score'].mean():.2f}")
print(f"  Median Score: {df['Lead Score'].median():.2f}")
print(f"  Min Score: {df['Lead Score'].min():.2f}")
print(f"  Max Score: {df['Lead Score'].max():.2f}")

print(f"\nLead Class Distribution:")
print(f"  🟢 High (70+): {(df['Lead Score'] >= 70).sum()} leads")
print(f"  🟡 Medium (40-69): {((df['Lead Score'] >= 40) & (df['Lead Score'] < 70)).sum()} leads")
print(f"  🔴 Low (<40): {(df['Lead Score'] < 40).sum()} leads")

# Display top 5 high-scoring leads
print(f"\n🏆 Top 5 High-Value Leads:")
top_leads = df.nlargest(5, 'Lead Score')[['Company Name', 'Country', 'Lead Score', 'Predicted Lead Class']]
for idx, row in top_leads.iterrows():
    print(f"  {idx+1}. {row['Company Name']} - {row['Country']} - Score: {row['Lead Score']:.0f} ({row['Predicted Lead Class']})")

# Save to CSV
df.to_csv('data/scored_leads.csv', index=False)
print("\n✅ Scoring complete! Saved to 'data/scored_leads.csv'")

# Optional: Show scoring breakdown for first few companies
print("\n" + "="*50)
print("Sample Scoring Breakdown (First 3 companies):")
print("="*50)
sample_cols = ['Company Name', 'Country', 'website_exists', 'contact_form', 
               'services_count', 'lead_quality_score', 'Lead Score', 'Predicted Lead Class']
if 'contact_page_exists' in df.columns:
    sample_cols.insert(5, 'contact_page_exists')
if 'industry_score' in df.columns:
    sample_cols.insert(7, 'industry_score')

print(df[sample_cols].head(3).to_string())