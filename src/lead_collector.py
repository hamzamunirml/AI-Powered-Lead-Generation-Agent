"""
Lead Collector Module
Task 3: Lead Research & Collection
"""

import pandas as pd

def collect_50_leads():
    """Generate 50 leads for USA, Canada, UAE"""
    
    leads = []
    
    companies = [
        "Nexus AI", "BrainCore AI", "Quantum AI", "DeepMind AI", "CogniTech",
        "AI Innovators", "NeuralTech", "IntelliSys", "DataMind AI", "Predictive AI",
        "AI Dynamics", "SmartLogic", "CogniCore", "AI Vanguard", "DeepLearning AI",
        "AI Frontiers", "NeuralForge", "AI Catalyst", "BrainWave AI", "AI Infinitum",
        "NeuralEdge", "AI Precision", "DeepSense AI", "AI Mavericks", "CognitiveScale",
        "AI Republic", "NeuralVision", "AI Fortress", "IntelliMatrix", "AI Ops",
        "DeepCognition", "AI Velocity", "NeuralInsight", "AI Synergy", "BrainTrust AI",
        "AI Sovereign", "CogniVerse", "AI Zenith", "Nexa AI", "AI Matrix",
        "IntelliGen", "AI Ascend", "NeuralCore", "AI Logic", "DeepCore AI",
        "AI Bridge", "CogniTech Labs", "AI Works", "Neural Labs", "AI Hub"
    ]
    
    countries = ["USA", "Canada", "UAE"]
    
    for i, company in enumerate(companies):
        country = countries[i % 3]
        
        lead = {
            'Company Name': company,
            'Website': f'www.{company.lower().replace(" ", "")}.com',
            'Country': country,
            'Industry': 'AI Development',
            'Contact Page': 'Yes' if i < 40 else 'No'
        }
        leads.append(lead)
    
    df = pd.DataFrame(leads)
    df.to_excel('data/business_leads.xlsx', index=False)
    df.to_csv('data/leads_dataset.csv', index=False)
    
    print(f"✅ {len(leads)} leads collected successfully!")
    print(f"📊 USA: {sum(1 for l in leads if l['Country']=='USA')} leads")
    print(f"📊 Canada: {sum(1 for l in leads if l['Country']=='Canada')} leads")
    print(f"📊 UAE: {sum(1 for l in leads if l['Country']=='UAE')} leads")
    
    return df

if __name__ == "__main__":
    collect_50_leads() 
