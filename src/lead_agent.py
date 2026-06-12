"""
Lead Recommendation Agent
Internship Program - Machine Learning Track

This agent loads the trained model and provides lead quality predictions
with actionable recommendations for the sales team.
"""

import pandas as pd
import numpy as np
import joblib
import os
import sys
import warnings
warnings.filterwarnings('ignore')


class LeadRecommendationAgent:
    """
    Lead Recommendation Agent for predicting lead quality and providing recommendations
    
    The agent predicts lead categories: High, Medium, Low
    and provides appropriate recommendations based on the prediction.
    """
    
    def __init__(self, model_path=None):
        """
        Initialize the Lead Recommendation Agent
        
        Args:
            model_path: Optional custom path to model files
        """
        # Initialize variables
        self.model = None
        self.label_encoder = None
        self.features = None
        
        # Define feature columns (as per requirements)
        self.feature_columns = ['website_exists', 'contact_form', 'services_count', 'country_score']
        
        # Set model paths (looking in notebooks/saved_models folder)
        if model_path:
            self.model_dir = model_path
        else:
            # Try multiple possible paths
            possible_paths = [
                'notebooks/saved_models',
                './notebooks/saved_models',
                '../notebooks/saved_models',
                'saved_models',
                './saved_models'
            ]
            
            self.model_dir = None
            for path in possible_paths:
                if os.path.exists(path):
                    self.model_dir = path
                    break
            
            if self.model_dir is None:
                # Check if notebooks folder exists
                if os.path.exists('notebooks'):
                    notebooks_files = os.listdir('notebooks')
                    if 'saved_models' in notebooks_files:
                        self.model_dir = 'notebooks/saved_models'
                    elif 'saved_models' in os.listdir('.'):
                        self.model_dir = 'saved_models'
        
        # Load all required artifacts
        self.load_model()
        self.load_label_encoder()
        self.load_features()
        
        # Check if everything loaded successfully
        self.is_ready = all([self.model is not None, self.label_encoder is not None, self.features is not None])
        
        if self.is_ready:
            print("\n" + "="*60)
            print("✅ LEAD RECOMMENDATION AGENT INITIALIZED SUCCESSFULLY")
            print("="*60)
            print(f"   Model: {type(self.model).__name__}")
            print(f"   Features: {self.features}")
            print(f"   Classes: {self.label_encoder.classes_}")
            print("="*60)
        else:
            print("\n⚠️ Agent initialized with missing components. Some features may not work.")
    
    def load_model(self):
        """Load the trained model from best_lead_model.pkl"""
        model_file = os.path.join(self.model_dir, 'best_lead_model.pkl') if self.model_dir else 'best_lead_model.pkl'
        
        try:
            if os.path.exists(model_file):
                self.model = joblib.load(model_file)
                print(f"✅ Model loaded successfully from: {model_file}")
                print(f"   Model type: {type(self.model).__name__}")
            else:
                print(f"❌ Model file not found: {model_file}")
                print(f"   Searching in: {self.model_dir}")
                self.model = None
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            self.model = None
    
    def load_label_encoder(self):
        """Load the label encoder from label_encoder.pkl"""
        encoder_file = os.path.join(self.model_dir, 'label_encoder.pkl') if self.model_dir else 'label_encoder.pkl'
        
        try:
            if os.path.exists(encoder_file):
                self.label_encoder = joblib.load(encoder_file)
                print(f"✅ Label encoder loaded from: {encoder_file}")
                print(f"   Classes: {self.label_encoder.classes_}")
            else:
                print(f"❌ Label encoder not found: {encoder_file}")
                self.label_encoder = None
        except Exception as e:
            print(f"❌ Error loading label encoder: {e}")
            self.label_encoder = None
    
    def load_features(self):
        """Load the feature names from feature_names.pkl"""
        features_file = os.path.join(self.model_dir, 'feature_names.pkl') if self.model_dir else 'feature_names.pkl'
        
        try:
            if os.path.exists(features_file):
                self.features = joblib.load(features_file)
                print(f"✅ Feature names loaded from: {features_file}")
                print(f"   Features: {self.features}")
            else:
                print(f"⚠️ Feature names file not found: {features_file}")
                print(f"   Using default features: {self.feature_columns}")
                self.features = self.feature_columns
        except Exception as e:
            print(f"❌ Error loading features: {e}")
            self.features = self.feature_columns
    
    def predict_lead_quality(self, website_exists, contact_form, services_count, country_score):
        """
        Predict lead quality category
        
        Args:
            website_exists: 0 or 1 (1 = has website)
            contact_form: 0 or 1 (1 = has contact form)
            services_count: integer (number of services/features)
            country_score: integer (1, 2, or 3 - country ranking)
        
        Returns:
            str: 'High', 'Medium', or 'Low'
        """
        if not self.is_ready:
            print("⚠️ Agent not ready. Using fallback prediction.")
            return self._fallback_prediction(website_exists, contact_form, services_count, country_score)
        
        try:
            # Create input dataframe
            input_data = pd.DataFrame([[
                website_exists, 
                contact_form, 
                services_count, 
                country_score
            ]], columns=self.features)
            
            # Make prediction
            prediction_encoded = self.model.predict(input_data)[0]
            
            # Decode to original label
            prediction = self.label_encoder.inverse_transform([prediction_encoded])[0]
            
            return prediction
            
        except Exception as e:
            print(f"❌ Prediction error: {e}")
            print("Using fallback prediction...")
            return self._fallback_prediction(website_exists, contact_form, services_count, country_score)
    
    def _fallback_prediction(self, website_exists, contact_form, services_count, country_score):
        """
        Fallback rule-based prediction when ML model fails
        
        Args:
            Same as predict_lead_quality
        
        Returns:
            str: 'High', 'Medium', or 'Low'
        """
        # Calculate score
        score = 0
        if website_exists == 1:
            score += 35
        if contact_form == 1:
            score += 35
        score += min(services_count, 5) * 10
        score += country_score * 6.67
        
        # Determine category
        if score >= 75:
            return 'High'
        elif score >= 45:
            return 'Medium'
        else:
            return 'Low'
    
    def get_recommendation(self, prediction):
        """
        Generate recommendation message based on prediction
        
        As per requirements:
        - High: Priority Lead - Contact within 24 hours
        - Medium: Potential Opportunity - Add to nurture campaign
        - Low: Low Priority - Monitor for future engagement
        
        Args:
            prediction: str - 'High', 'Medium', or 'Low'
        
        Returns:
            str: Recommendation message
        """
        recommendations = {
            'High': "Priority Lead - Contact within 24 hours",
            'Medium': "Potential Opportunity - Add to nurture campaign",
            'Low': "Low Priority - Monitor for future engagement"
        }
        return recommendations.get(prediction, "Unknown - Please check lead data")
    
    def get_action_priority(self, prediction):
        """
        Get action priority based on prediction
        
        Args:
            prediction: str - 'High', 'Medium', or 'Low'
        
        Returns:
            dict: Priority details
        """
        priorities = {
            'High': {'priority': '🔴 HIGH', 'urgency': 'Immediate', 'color': 'red'},
            'Medium': {'priority': '🟡 MEDIUM', 'urgency': 'This Week', 'color': 'yellow'},
            'Low': {'priority': '🟢 LOW', 'urgency': 'Next Month', 'color': 'green'}
        }
        return priorities.get(prediction, {'priority': '⚪ UNKNOWN', 'urgency': 'TBD', 'color': 'gray'})
    
    def analyze_lead(self, company_name, website_exists, contact_form, services_count, country_score):
        """
        Complete analysis for a single lead
        
        This is the main method to be used for lead analysis.
        
        Args:
            company_name: str - Name of the company
            website_exists: 0 or 1
            contact_form: 0 or 1
            services_count: int
            country_score: 1, 2, or 3
        
        Returns:
            dict: Complete analysis result
        """
        # Predict lead quality
        prediction = self.predict_lead_quality(
            website_exists, 
            contact_form, 
            services_count, 
            country_score
        )
        
        # Get recommendation
        recommendation = self.get_recommendation(prediction)
        
        # Get action priority
        priority_info = self.get_action_priority(prediction)
        
        return {
            'company_name': company_name,
            'prediction': prediction,
            'recommendation': recommendation,
            'priority': priority_info['priority'],
            'urgency': priority_info['urgency'],
            'features_used': {
                'website_exists': website_exists,
                'contact_form': contact_form,
                'services_count': services_count,
                'country_score': country_score
            }
        }
    
    def analyze_multiple_leads(self, leads_data):
        """
        Analyze multiple leads
        
        Args:
            leads_data: Can be either:
                - List of dictionaries with keys: company_name, website_exists, contact_form, 
                  services_count, country_score
                - List of tuples/lists: (company_name, website_exists, contact_form, 
                  services_count, country_score)
                - pandas DataFrame with required columns
        
        Returns:
            list: List of analysis results
        """
        results = []
        
        # Handle DataFrame input
        if isinstance(leads_data, pd.DataFrame):
            for _, row in leads_data.iterrows():
                result = self.analyze_lead(
                    row.get('Company Name', row.get('company_name', 'Unknown')),
                    row.get('website_exists', 0),
                    row.get('contact_form', 0),
                    row.get('services_count', 0),
                    row.get('country_score', 1)
                )
                results.append(result)
        
        # Handle list of dictionaries
        elif isinstance(leads_data, list) and len(leads_data) > 0:
            for lead in leads_data:
                if isinstance(lead, dict):
                    result = self.analyze_lead(
                        lead.get('company_name', lead.get('Company Name', 'Unknown')),
                        lead.get('website_exists', 0),
                        lead.get('contact_form', 0),
                        lead.get('services_count', 0),
                        lead.get('country_score', 1)
                    )
                elif isinstance(lead, (list, tuple)) and len(lead) >= 5:
                    result = self.analyze_lead(lead[0], lead[1], lead[2], lead[3], lead[4])
                else:
                    print(f"⚠️ Skipping invalid lead format: {lead}")
                    continue
                results.append(result)
        
        return results
    
    def predict_from_dataframe(self, df, company_col='Company Name'):
        """
        Predict leads from a DataFrame
        
        Args:
            df: pandas DataFrame with columns: website_exists, contact_form, 
                services_count, country_score
            company_col: Column name for company names
        
        Returns:
            pandas DataFrame with predictions added
        """
        result_df = df.copy()
        predictions = []
        recommendations = []
        priorities = []
        
        for idx, row in df.iterrows():
            pred = self.predict_lead_quality(
                row.get('website_exists', 0),
                row.get('contact_form', 0),
                row.get('services_count', 0),
                row.get('country_score', 1)
            )
            rec = self.get_recommendation(pred)
            priority = self.get_action_priority(pred)['priority']
            
            predictions.append(pred)
            recommendations.append(rec)
            priorities.append(priority)
        
        result_df['Predicted_Lead_Class'] = predictions
        result_df['Recommendation'] = recommendations
        result_df['Priority'] = priorities
        
        return result_df
    
    def get_prediction_probability(self, website_exists, contact_form, services_count, country_score):
        """
        Get prediction probabilities if model supports it
        
        Args:
            Same as predict_lead_quality
        
        Returns:
            dict: Probabilities for each class
        """
        if not self.is_ready or not hasattr(self.model, 'predict_proba'):
            return None
        
        try:
            input_data = pd.DataFrame([[
                website_exists, contact_form, services_count, country_score
            ]], columns=self.features)
            
            probabilities = self.model.predict_proba(input_data)[0]
            
            return {
                self.label_encoder.classes_[i]: probabilities[i] 
                for i in range(len(self.label_encoder.classes_))
            }
        except Exception as e:
            print(f"❌ Error getting probabilities: {e}")
            return None


# ============================================
# TESTING AND DEMONSTRATION
# ============================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("LEAD RECOMMENDATION AGENT - TESTING & DEMONSTRATION")
    print("="*70)
    
    # Initialize agent
    agent = LeadRecommendationAgent()
    
    if not agent.is_ready:
        print("\n⚠️ Agent not properly initialized. Please check model files.")
        print("Expected files in 'notebooks/saved_models/':")
        print("  - best_lead_model.pkl")
        print("  - label_encoder.pkl")
        print("  - feature_names.pkl")
        sys.exit(1)
    
    # Test Single Lead
    print("\n" + "="*60)
    print("TEST 1: Single Lead Analysis")
    print("="*60)
    
    test_result = agent.analyze_lead(
        company_name="ABC Solutions",
        website_exists=1,
        contact_form=1,
        services_count=5,
        country_score=3
    )
    
    print(f"\nCompany: {test_result['company_name']}")
    print(f"Prediction: {test_result['prediction']}")
    print(f"Priority: {test_result['priority']}")
    print(f"Urgency: {test_result['urgency']}")
    print(f"Recommendation: {test_result['recommendation']}")
    
    # Test Multiple Leads (as per Task 4 requirements)
    print("\n" + "="*60)
    print("TEST 2: Multiple Leads Analysis (10 Sample Leads)")
    print("="*60)
    
    sample_leads = [
        ("ABC Solutions", 1, 1, 5, 3),
        ("Tech Corp", 1, 1, 4, 2),
        ("StartUp Inc", 1, 0, 3, 2),
        ("Digital Dynamics", 1, 1, 3, 1),
        ("Web Innovators", 1, 0, 2, 2),
        ("Small Biz", 0, 0, 1, 1),
        ("Global Tech", 1, 1, 5, 3),
        ("Local Services", 0, 1, 2, 1),
        ("AI Solutions", 1, 1, 4, 2),
        ("Data Systems", 1, 0, 3, 2)
    ]
    
    results = agent.analyze_multiple_leads(sample_leads)
    
    print("\n{:<25} {:^10} {:^45}".format("Company Name", "Prediction", "Recommendation"))
    print("-"*80)
    
    for result in results:
        print("{:<25} {:^10} {:45}".format(
            result['company_name'],
            result['prediction'],
            result['recommendation']
        ))
    
    # Test with DataFrame
    print("\n" + "="*60)
    print("TEST 3: DataFrame Input")
    print("="*60)
    
    test_df = pd.DataFrame([
        {"Company Name": "Test Company 1", "website_exists": 1, "contact_form": 1, "services_count": 5, "country_score": 3},
        {"Company Name": "Test Company 2", "website_exists": 0, "contact_form": 0, "services_count": 1, "country_score": 1}
    ])
    
    result_df = agent.predict_from_dataframe(test_df)
    print("\nResults with Predictions:")
    print(result_df[['Company Name', 'Predicted_Lead_Class', 'Priority', 'Recommendation']].to_string(index=False))
    
    # Get Probabilities
    print("\n" + "="*60)
    print("TEST 4: Prediction Probabilities")
    print("="*60)
    
    probabilities = agent.get_prediction_probability(1, 1, 5, 3)
    if probabilities:
        print("\nPrediction Probabilities:")
        for lead_class, prob in probabilities.items():
            print(f"  {lead_class}: {prob:.2%}")
    
    # Final Summary
    print("\n" + "="*70)
    print("✅ LEAD RECOMMENDATION AGENT - READY FOR USE")
    print("="*70)
    print("\n📋 Agent Features:")
    print("   ✅ Loads trained model from notebooks/saved_models/")
    print("   ✅ Uses best_lead_model.pkl, label_encoder.pkl, feature_names.pkl")
    print("   ✅ Predicts: High, Medium, Low")
    print("   ✅ Provides recommendations as per requirements")
    print("   ✅ Handles single and multiple leads")
    print("   ✅ Supports DataFrame input")
    print("   ✅ Fallback prediction if model fails")
    print("\n💡 Usage Example:")
    print("   from lead_agent import LeadRecommendationAgent")
    print("   agent = LeadRecommendationAgent()")
    print("   result = agent.analyze_lead('Company Name', 1, 1, 5, 3)")
    print("   print(result['prediction'], result['recommendation'])")
    print("="*70)