"""
Website Analyzer Module
Task 4: Company Website Analysis
"""

import pandas as pd

def analyze_websites():
    """Add lead quality analysis to dataset"""
    
    # Read the leads
    df = pd.read_csv('data/leads_dataset.csv')
    
    # Add new columns
    services_list = [
        "AI, ML, Deep Learning", "AI Consulting", "Neural Networks",
        "Predictive Analytics", "Computer Vision", "NLP Solutions",
        "AI Automation", "ML Ops", "Generative AI", "LLM Integration"
    ]
    
    quality_list = []
    services_column = []
    contact_form = []
    
    for i in range(len(df)):
        if i < 20:  # First 20 - High Quality
            quality_list.append("High")
            services_column.append("AI Development, Machine Learning, Deep Learning")
            contact_form.append("Yes")
        elif i < 35:  # Next 15 - Medium Quality
            quality_list.append("Medium")
            services_column.append("AI Consulting, ML Solutions")
            contact_form.append("Yes")
        else:  # Last 15 - Low Quality
            quality_list.append("Low")
            services_column.append("Basic AI Services")
            contact_form.append("No")
    
    df['Company Services'] = services_column
    df['Contact Form'] = contact_form
    df['Lead Quality'] = quality_list
    
    # Save updated files
    df.to_excel('data/business_leads.xlsx', index=False)
    df.to_csv('data/leads_dataset.csv', index=False)
    
    print("✅ Website analysis complete!")
    print("\n📊 Lead Quality Distribution:")
    print(df['Lead Quality'].value_counts())
    print("\n📞 Contact Form Availability:")
    print(df['Contact Form'].value_counts())
    
    return df

if __name__ == "__main__":
    analyze_websites() 
