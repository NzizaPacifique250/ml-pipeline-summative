import streamlit as st
import requests
import json
import os
import matplotlib.pyplot as plt

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(page_title="ML Pipeline App", layout="wide")

st.title("🐱🐶 Cat vs Dog ML Pipeline")

tab1, tab2, tab3, tab4 = st.tabs(["Prediction", "Visualizations", "Retraining", "Dashboard"])

with tab1:
    st.header("Make a Prediction")
    st.write("Upload an image to see if it's a cat or a dog.")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
    
    if uploaded_file is not None:
        st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)
        if st.button("Predict"):
            try:
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                response = requests.post(f"{API_URL}/predict", files=files)
                if response.status_code == 200:
                    result = response.json().get("result")
                    st.success(f"**Prediction:** {result['prediction']}")
                    st.info(f"**Confidence:** {result['confidence'] * 100:.2f}%")
                else:
                    st.error(f"Error {response.status_code}: {response.text}")
            except Exception as e:
                st.error(f"Failed to connect to API: {e}")

with tab2:
    st.header("Data Visualizations")
    st.write("Exploratory Data Analysis metrics for our dataset.")
    
    if st.button("Load Dataset Statistics"):
        train_dir = "data/train"
        val_dir = "data/validation"
        
        # Simple counts
        def get_counts(directory):
            try:
                cats = len(os.listdir(os.path.join(directory, "cats")))
                dogs = len(os.listdir(os.path.join(directory, "dogs")))
                return cats, dogs
            except:
                return 0, 0
                
        train_cats, train_dogs = get_counts(train_dir)
        val_cats, val_dogs = get_counts(val_dir)
        
        st.subheader("Dataset Distribution")
        fig, ax = plt.subplots(1, 2, figsize=(12, 5))
        
        ax[0].bar(["Cats", "Dogs"], [train_cats, train_dogs], color=["blue", "orange"])
        ax[0].set_title("Training Data")
        ax[0].set_ylabel("Number of Submissions")
        
        ax[1].bar(["Cats", "Dogs"], [val_cats, val_dogs], color=["blue", "orange"])
        ax[1].set_title("Validation Data")
        
        st.pyplot(fig)
        
        st.write("These metrics represent three core insights: the balance of the target variables (Classes), the absolute training volume, and the validation split sizes ensuring adequate evaluation logic.")

with tab3:
    st.header("Retrain the Model")
    st.write("Upload new images to augment the dataset and retrain the classifier dynamically.")
    
    label_choice = st.radio("Label for uploaded images:", ("cats", "dogs"))
    bulk_files = st.file_uploader("Upload multiple training images...", type=["jpg", "png", "jpeg"], accept_multiple_files=True)
    
    if st.button("Upload Data"):
        if bulk_files:
            try:
                files_payload = [("files", (f.name, f.getvalue(), f.type)) for f in bulk_files]
                data_payload = {"label": label_choice}
                response = requests.post(f"{API_URL}/upload_data", data=data_payload, files=files_payload)
                if response.status_code == 200:
                    st.success(f"Successfully uploaded {len(bulk_files)} images for {label_choice}!")
                else:
                    st.error(f"Error: {response.text}")
            except Exception as e:
                st.error(f"Failed to upload: {e}")
        else:
            st.warning("Please select files first.")
            
    st.markdown("---")
    if st.button("Trigger Retraining Pipeline", type="primary"):
        try:
            res = requests.post(f"{API_URL}/retrain")
            if res.status_code == 200:
                st.success("Retraining task has been dispatched successfully! Check logs for progress.")
            else:
                st.error(f"Error triggering retraining: {res.text}")
        except Exception as e:
            st.error(f"Failed to contact API: {e}")

with tab4:
    st.header("Model Up-Time Dashboard")
    try:
        res = requests.get(f"{API_URL}/health")
        if res.status_code == 200:
            data = res.json()
            st.metric("API Status", "🟢 Online")
            st.metric("Uptime (seconds)", f"{data['uptime_seconds']}")
        else:
            st.metric("API Status", "🔴 Error")
    except:
        st.metric("API Status", "🔴 Offline - Server Unreachable")
