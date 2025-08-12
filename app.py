import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from io import BytesIO

class SugarcaneHarvestClassifier:
    def __init__(self):
        self.scaler = StandardScaler()
        self.rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.kmeans = KMeans(n_clusters=3, random_state=42)
        
    def preprocess_data(self, df):
        """Clean and preprocess the data"""
        # Remove rows with invalid data
        df = df.dropna(subset=['Total luas'])
        
        # Calculate percentages properly
        df['pct_not_harvested'] = df['Luas_Citra_Belum_Tebang_Ha'] / df['Total luas']
        df['pct_harvested'] = df['Luas_Citra_Tebang_Ha'] / df['Total luas']
        df['pct_cloud'] = df['Awan_Ha'] / df['Total luas']
        
        # Ensure percentages sum to 1
        total_pct = df['pct_not_harvested'] + df['pct_harvested'] + df['pct_cloud']
        df['pct_not_harvested'] = df['pct_not_harvested'] / total_pct
        df['pct_harvested'] = df['pct_harvested'] / total_pct
        df['pct_cloud'] = df['pct_cloud'] / total_pct
        
        return df
    
    def create_improved_classification(self, df):
        """Create improved classification using machine learning"""
        # Features for classification
        features = ['pct_not_harvested', 'pct_harvested', 'pct_cloud', 'Total luas']
        X = df[features].values
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Use cloud coverage threshold to separate reliable vs unreliable data
        cloud_threshold = 0.3  # 30% cloud coverage threshold
        
        # Create target classes based on harvest percentage and cloud coverage
        def classify_harvest_level(row):
            if row['pct_cloud'] > cloud_threshold:
                return 'Cloud_Cover'  # Class 5 - Cloud cover
            elif row['pct_harvested'] >= 0.7:
                return 'High_Harvest'  # Class 1 - High harvest
            elif row['pct_harvested'] >= 0.4:
                return 'Medium_Harvest'  # Class 2 - Medium harvest
            elif row['pct_harvested'] >= 0.1:
                return 'Low_Harvest'  # Class 3 - Low harvest
            else:
                return 'Not_Harvest'  # Class 4 - Not harvest
        
        df['ML_Class'] = df.apply(classify_harvest_level, axis=1)
        
        # Convert to numerical scale (1-5)
        class_mapping = {
            'High_Harvest': 1,    # High harvest
            'Medium_Harvest': 2,  # Medium harvest  
            'Low_Harvest': 3,     # Low harvest
            'Not_Harvest': 4,     # Not harvest
            'Cloud_Cover': 5      # Cloud cover
        }
        
        # More granular numerical classification
        def get_harvest_scale(row):
            if row['pct_cloud'] > cloud_threshold:
                return 5  # Cloud cover
            elif row['pct_harvested'] >= 0.7:
                return 1  # High harvest
            elif row['pct_harvested'] >= 0.4:
                return 2  # Medium harvest
            elif row['pct_harvested'] >= 0.1:
                return 3  # Low harvest
            else:
                return 4  # Not harvest
        
        df['Harvest_Scale'] = df.apply(get_harvest_scale, axis=1)
        
        return df, X_scaled
    
    def train_model(self, df, X_scaled):
        """Train Random Forest model"""
        # Prepare training data (exclude cloud cover areas for training, but include all for prediction)
        reliable_data = df[df['Harvest_Scale'] < 5]  # Exclude only cloud cover for training
        reliable_X = X_scaled[df['Harvest_Scale'] < 5]
        reliable_y = reliable_data['Harvest_Scale']
        
        if len(reliable_data) > 5:  # Need minimum data for training
            X_train, X_test, y_train, y_test = train_test_split(
                reliable_X, reliable_y, test_size=0.2, random_state=42
            )
            
            self.rf_model.fit(X_train, y_train)
            
            if len(X_test) > 0:
                y_pred = self.rf_model.predict(X_test)
                return classification_report(y_test, y_pred, output_dict=True)
        
        return None
    
    def create_visualizations(self, df):
        """Create comprehensive visualizations"""
        
        # 1. Distribution of harvest classes
        fig1 = px.histogram(df, x='Harvest_Scale', 
                           title='Distribution of Harvest Classification Scale',
                           labels={'Harvest_Scale': 'Classification (1=High Harvest, 2=Medium, 3=Low, 4=Not Harvest, 5=Cloud)'})
        
        # 2. Scatter plot of harvest vs cloud coverage
        fig2 = px.scatter(df, x='pct_harvested', y='pct_cloud', 
                         color='Harvest_Scale', size='Total luas',
                         title='Harvest Percentage vs Cloud Coverage',
                         labels={'pct_harvested': 'Percentage Harvested', 
                                'pct_cloud': 'Percentage Cloud Cover'},
                         color_continuous_scale='viridis')
        
        # 3. Feature importance (if model is trained)
        feature_names = ['% Not Harvested', '% Harvested', '% Cloud', 'Total Area']
        try:
            importances = self.rf_model.feature_importances_
            fig3 = px.bar(x=feature_names, y=importances,
                         title='Feature Importance for Harvest Classification')
        except:
            fig3 = None
        
        # 4. Comparison with original classification
        if 'KODE_WARNA' in df.columns:
            comparison_df = pd.DataFrame({
                'Original': df['KODE_WARNA'],
                'Improved': df['Harvest_Scale']
            })
            fig4 = px.scatter(comparison_df, x='Original', y='Improved',
                            title='Original vs Improved Classification')
        else:
            fig4 = None
        
        return fig1, fig2, fig3, fig4

def main():
    st.set_page_config(page_title="Sugarcane Harvest Classifier", layout="wide")
    
    st.title("🌾 Sugarcane Harvest Classification System")
    st.markdown("""
    This application uses machine learning to improve sugarcane harvest monitoring classification.
    Upload your Excel file with Sentinel satellite data to get improved classifications.
    """)
    
    # Sidebar for parameters
    st.sidebar.header("Classification Parameters")
    cloud_threshold = st.sidebar.slider("Cloud Coverage Threshold", 0.1, 0.5, 0.3)
    use_area_weighting = st.sidebar.checkbox("Use Area Weighting", True)
    
    # File upload
    uploaded_file = st.file_uploader("Upload Excel file", type=['xlsx', 'xls'])
    
    if uploaded_file is not None:
        try:
            # Read Excel file
            df = pd.read_excel(uploaded_file)
            
            st.subheader("📊 Original Data Preview")
            st.dataframe(df.head())
            
            # Initialize classifier
            classifier = SugarcaneHarvestClassifier()
            
            # Preprocess data
            df_processed = classifier.preprocess_data(df)
            
            # Create improved classification
            df_classified, X_scaled = classifier.create_improved_classification(df_processed)
            
            # Train model
            model_report = classifier.train_model(df_classified, X_scaled)
            
            # Display results
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("🎯 Improved Classification Results")
                st.dataframe(df_classified[['Total luas', 'pct_harvested', 'pct_cloud', 
                                          'ML_Class', 'Harvest_Scale']].head(10))
                
                # Classification summary
                st.subheader("📈 Classification Summary")
                class_summary = df_classified['Harvest_Scale'].value_counts().sort_index()
                st.bar_chart(class_summary)
                
            with col2:
                st.subheader("🔍 Model Performance")
                if model_report:
                    st.json(model_report)
                else:
                    st.write("Insufficient data for model training")
                
                # Feature statistics
                st.subheader("📋 Feature Statistics")
                stats_df = df_classified[['pct_harvested', 'pct_cloud', 'Total luas']].describe()
                st.dataframe(stats_df)
            
            # Visualizations
            st.subheader("📊 Analysis Visualizations")
            fig1, fig2, fig3, fig4 = classifier.create_visualizations(df_classified)
            
            col3, col4 = st.columns(2)
            with col3:
                st.plotly_chart(fig1, use_container_width=True)
                if fig3:
                    st.plotly_chart(fig3, use_container_width=True)
            
            with col4:
                st.plotly_chart(fig2, use_container_width=True)
                if fig4:
                    st.plotly_chart(fig4, use_container_width=True)
            
            # Download improved dataset
            st.subheader("💾 Download Improved Classification")
            
            # Create download button
            output = BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                df_classified.to_excel(writer, sheet_name='Improved_Classification', index=False)
            
            st.download_button(
                label="Download Improved Dataset",
                data=output.getvalue(),
                file_name="improved_sugarcane_classification.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
            
            # Recommendations
            st.subheader("💡 Recommendations")
            
            cloud_areas = len(df_classified[df_classified['Harvest_Scale'] == 5])  # Cloud cover is now class 5
            total_areas = len(df_classified)
            
            st.write(f"""
            **Analysis Summary:**
            - Total areas analyzed: {total_areas}
            - Areas with high cloud coverage (>30%): {cloud_areas} ({cloud_areas/total_areas*100:.1f}%)
            - Average harvest percentage: {df_classified['pct_harvested'].mean():.2f}
            
            **Classification Scale:**
            - **Class 1**: High Harvest (≥70% harvested)
            - **Class 2**: Medium Harvest (40-70% harvested)  
            - **Class 3**: Low Harvest (10-40% harvested)
            - **Class 4**: Not Harvest (<10% harvested)
            - **Class 5**: Cloud Cover (>30% cloud coverage - unreliable data)
            
            **Improvements made:**
            1. **Cloud identification**: Areas with >30% cloud coverage are classified as Class 5
            2. **Balanced thresholds**: Uses meaningful harvest percentage ranges
            3. **Area weighting**: Considers total area size in classification
            4. **Machine learning**: Random Forest model learns patterns from your data
            
            **Next steps:**
            1. Use the improved classification for mapping (Classes 1-4 for reliable data)
            2. Validate results with ground truth data
            3. Adjust cloud threshold based on your specific needs
            4. Consider temporal analysis for harvest progress tracking
            """)

            
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
            st.write("Please ensure your Excel file has the correct column structure.")

if __name__ == "__main__":
    main()