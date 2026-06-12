import pandas as pd

# 1. File load karo
df = pd.read_excel('data/business_leads.xlsx')

# 2. Duplicate companies remove karo
initial_count = len(df)
df = df.drop_duplicates(subset=['Company Name'])
print(f"Removed {initial_count - len(df)} duplicates")

# 3. Jahan website missing hai wahan remove karo
df = df.dropna(subset=['Website'])
print(f"Removed rows with missing website")

# 4. Country names standardize karo
df['Country'] = df['Country'].replace({
    'USA': 'United States',
    'UAE': 'United Arab Emirates',
    'Canada': 'Canada'
})

# 5. Contact Page values standardize karo
df['Contact Page'] = df['Contact Page'].apply(lambda x: 'Yes' if x == 'Yes' else 'No')

# 6. Lead Quality spelling check
df['Lead Quality'] = df['Lead Quality'].replace({
    'High': 'High',
    'Medium': 'Medium', 
    'Low': 'Low'
})

# 7. Save cleaned file
df.to_csv('data/cleaned_leads.csv', index=False)

print(f"Final cleaned records: {len(df)}")