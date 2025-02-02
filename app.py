import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
import joblib

def load_data():
    """Load and preprocess the data"""
    data = pd.read_csv('cancer.csv')
    data_cleaned = data.drop(["id", "Unnamed: 32"], axis=1)
    
    # Encode diagnosis
    le = LabelEncoder()
    data_cleaned['diagnosis'] = le.fit_transform(data_cleaned['diagnosis'])
    
    return data_cleaned

def train_model(data_cleaned):
    """Train and save the model"""
    X = data_cleaned.drop('diagnosis', axis=1)
    y = data_cleaned['diagnosis']
    
    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)
    
    # Save model and feature names
    joblib.dump(model, 'model.pkl')
    joblib.dump(X.columns.tolist(), 'features.pkl')
    
    return model, X.columns.tolist()

def main():
    st.title("🏥 Cancer Detection Assistant")
    
    # Load data and train model
    try:
        data_cleaned = load_data()
        try:
            # Try to load existing model and features
            model = joblib.load('model.pkl')
            features = joblib.load('features.pkl')
        except:
            # If loading fails, train new model
            # st.info("Training new model...")
            # model, features = train_model(data_cleaned)
            # st.success("Model training complete!")
            print()
        
        # Educational section
        st.markdown("---")
        col1, col2 = st.columns(2)
        with col1:
            st.info("### Benign Tumor\n"
                    "- NOT cancerous\n"
                    "- Doesn't spread to other parts\n"
                    "- Usually removable\n"
                    "- Rarely life threatening")
        
        with col2:
            st.error("### Malignant Tumor\n"
                     "- IS cancerous\n"
                     "- Can spread to other parts\n"
                     "- Requires immediate attention\n"
                     "- Needs quick treatment")
        
        # Input section
        st.markdown("---")
        st.header("Enter Measurements")
        
        # Create organized input sections
        user_inputs = {}
        
        # Create tabs for different measurement categories
        tabs = st.tabs(["Size Measurements", "Texture Features", "Other Features"])
        
        # Get mean values from cleaned data for defaults
        X = data_cleaned.drop('diagnosis', axis=1)
        
        with tabs[0]:
            col1, col2 = st.columns(2)
            with col1:
                for feature in ['radius_mean', 'perimeter_mean', 'area_mean']:
                    user_inputs[feature] = st.number_input(
                        f"{feature.replace('_', ' ').title()}", 
                        value=float(X[feature].mean())
                    )
            with col2:
                for feature in ['radius_worst', 'perimeter_worst', 'area_worst']:
                    user_inputs[feature] = st.number_input(
                        f"{feature.replace('_', ' ').title()}", 
                        value=float(X[feature].mean())
                    )
        
        with tabs[1]:
            col1, col2 = st.columns(2)
            with col1:
                for feature in ['texture_mean', 'smoothness_mean', 'compactness_mean']:
                    user_inputs[feature] = st.number_input(
                        f"{feature.replace('_', ' ').title()}", 
                        value=float(X[feature].mean())
                    )
            with col2:
                for feature in ['texture_worst', 'smoothness_worst', 'compactness_worst']:
                    user_inputs[feature] = st.number_input(
                        f"{feature.replace('_', ' ').title()}", 
                        value=float(X[feature].mean())
                    )
        
        with tabs[2]:
            # Add remaining features
            remaining_features = [f for f in features if f not in user_inputs]
            col1, col2 = st.columns(2)
            for i, feature in enumerate(remaining_features):
                with col1 if i % 2 == 0 else col2:
                    user_inputs[feature] = st.number_input(
                        f"{feature.replace('_', ' ').title()}", 
                        value=float(X[feature].mean())
                    )
        
        # Make prediction
        if st.button("Get Results", type="primary"):
            # Ensure we have all features in correct order
            input_vector = [user_inputs[feature] for feature in features]
            prediction = model.predict([input_vector])[0]
            
            st.markdown("---")
            if prediction == 0:
                st.success("### 🟢 Result: Likely Benign\n"
                          "The tumor characteristics suggest it is likely benign (non-cancerous).")
            else:
                st.error("### 🔴 Result: Likely Malignant\n"
                        "The tumor characteristics suggest it is likely malignant (cancerous).")

            st.warning("⚠️ **Important:** This is only a screening tool. Please consult with a "
                      "healthcare professional for proper diagnosis and treatment.")
            
    except FileNotFoundError:
        st.error("Error: Please make sure 'cancer.csv' is in the same directory as the application.")
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()