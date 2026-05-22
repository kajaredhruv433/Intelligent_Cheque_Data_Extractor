import streamlit as st
import os
import pandas as pd
import cv2
import time
from extractor import ChequeExtractor

# Set page config
st.set_page_config(page_title="Cheque Extractor", layout="wide")

st.title("🏦 Cheque Details Extractor")
st.markdown("Batch process cheque images from a folder.")

# Session State for Extractor
if 'extractor' not in st.session_state:
    st.session_state.extractor = None

# Input Folder
folder_path = st.text_input("📁 Enter Folder Path containing Images:", "")

# Load Model Button (Lazy Load)
@st.cache_resource
def load_extractor():
    return ChequeExtractor()

if st.button("🚀 Start Processing"):
    if not folder_path or not os.path.exists(folder_path):
        st.error("Invalid folder path!")
    else:
        with st.spinner("Loading Models..."):
            try:
                extractor = load_extractor()
            except Exception as e:
                st.error(f"Failed to load models: {e}")
                st.stop()
        
        # Get Images
        valid_exts = ('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.tif')
        files = [f for f in os.listdir(folder_path) if f.lower().endswith(valid_exts)]
        
        if not files:
            st.warning("No image files found in the directory.")
        else:
            st.info(f"Found {len(files)} images. Processing...")
            
            progress_bar = st.progress(0)
            results_list = []
            
            # Create a placeholder for live updates
            status_text = st.empty()
            table_placeholder = st.empty() # Placeholder for the dataframe
            metrics_placeholder = st.empty() 

            total_start_time = time.time()
            
            for i, filename in enumerate(files):
                img_start_time = time.time()
                file_path = os.path.join(folder_path, filename)
                status_text.text(f"Processing ({i+1}/{len(files)}): {filename}...")
                
                try:
                    data, _ = extractor.process_image(file_path)
                    data['Filename'] = filename # Add filename
                    
                    # Match Logic
                    amt_digit = data.get("Amount in Digit", "0")
                    amt_word = data.get("Amount in Word", "0")
                    
                    # Convert to float for comparison (handle potentially non-numeric strings safely)
                    try:
                         val_digit = float(amt_digit)
                    except:
                         val_digit = 0.0
                         
                    try:
                         val_word = float(amt_word)
                    except:
                         val_word = 0.0
                    
                    # Exact match or very close?
                    if val_digit == val_word and val_digit > 0:
                        data['Match'] = "Yes"
                    else:
                        data['Match'] = "No"

                    # Calculate time
                    img_end_time = time.time()
                    duration = img_end_time - img_start_time
                    data['Time (s)'] = f"{duration:.2f}"
                        
                    results_list.append(data)
                    
                    # LIVE UPDATE: Show table so far
                    df_live = pd.DataFrame(results_list)
                    # Reorder columns
                    cols = ['Filename', 'Match', 'Time (s)'] + [c for c in df_live.columns if c not in ['Filename', 'Match', 'Time (s)']]
                    df_live = df_live[cols]
                    table_placeholder.dataframe(df_live)

                    # Update metrics
                    avg_time = (time.time() - total_start_time) / (i + 1)
                    metrics_placeholder.markdown(f"**Avg Time/Image:** {avg_time:.2f}s | **Last Image:** {duration:.2f}s")
                    
                except Exception as e:
                    st.error(f"Error processing {filename}: {e}")
                
                progress_bar.progress((i + 1) / len(files))
            
            status_text.text("Processing Complete!")
            
            if results_list:
                df = pd.DataFrame(results_list)
                
                # Reorder columns: Filename first
                cols = ['Filename', 'Match', 'Time (s)'] + [c for c in df.columns if c not in ['Filename', 'Match', 'Time (s)']]
                df = df[cols]
                
                st.success("Analysis Complete!")
                # Final table update (in case anything missed, though live update covers it)
                table_placeholder.dataframe(df)
                
                # Save CSV
                folder_name = os.path.basename(os.path.normpath(folder_path))
                csv_filename = f"{folder_name}.csv"
                save_path = os.path.join(folder_path, csv_filename)
                
                df.to_csv(save_path, index=False)
                st.success(f"✅ Results saved to: **{save_path}**")
