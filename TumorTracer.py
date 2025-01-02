import streamlit as st
import torch
import cv2
import numpy as np
import segmentation_models_pytorch as smp

st.set_page_config(page_title="TumorTracer")

def load_model(model_path):
    model = smp.DeepLabV3Plus(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=1,
        classes=1,
    )

    state_dict = torch.load(model_path, map_location=torch.device('cpu'))
    new_state_dict = {}
    for k, v in state_dict.items():
        name = k.replace("module.", "") if k.startswith("module.") else k
        new_state_dict[name] = v
    
    model.load_state_dict(new_state_dict)
    model.eval()
    return model

def preprocess_image(img_array):
    img = cv2.resize(img_array, (256, 256))
    img = img / 255.0
    img = (img - 0.5)
    img = np.expand_dims(img, axis=0)
    img = np.expand_dims(img, axis=0)
    return torch.tensor(img, dtype=torch.float32)

def get_prediction(model, img_array):
    input_tensor = preprocess_image(img_array)
    with torch.no_grad():
        output = model(input_tensor)
        output = torch.sigmoid(output).squeeze().numpy()
    return (output > 0.5).astype(np.uint8)

def main():
    st.title("Brain Tumor MRI Segmentation")
    
    # Medical Disclaimer
    st.warning("""
        ⚠️ MEDICAL DISCLAIMER:
        
        This application is for educational and demonstration purposes only. 
        It should NOT be used for medical diagnosis or clinical decision-making. 
        Please consult qualified healthcare professionals for any medical concerns.
    """)
    
    st.write("Upload a brain MRI image to detect and segment tumors.")

    # Load model
    try:
        model = load_model("tumor_segmentation_model.pth")
        st.success("Model loaded successfully!")
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return

    # File uploader
    uploaded_file = st.file_uploader("Choose a brain MRI image...", type=["jpg", "jpeg", "png", "tif"])

    if uploaded_file is not None:
        try:
            # Convert uploaded file to image
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
            
            # Create columns for side-by-side display
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Original Image")
                st.image(img, caption="Input MRI", use_container_width=True)

            # Get prediction
            mask = get_prediction(model, img)
            
            # Scale mask for better visibility (0 to 255)
            mask_display = mask * 255
            
            with col2:
                st.subheader("Segmented Tumor")
                st.image(mask_display, caption="Predicted Tumor Mask", use_container_width=True)

            # Additional analysis
            if np.any(mask):
                tumor_percentage = (np.sum(mask) / mask.size) * 100
                st.write(f"Detected tumor region covers approximately {tumor_percentage:.2f}% of the image")
                
                # Add overlay visualization
                overlay = img.copy()
                overlay = cv2.resize(overlay, (256, 256))  # Resize to match mask size
                overlay[mask == 1] = 255  # Highlight tumor regions
                
                st.subheader("Overlay Visualization")
                st.image(overlay, caption="Tumor Overlay", use_container_width=True)
            else:
                st.write("No tumor detected in the image")

        except Exception as e:
            st.error(f"Error processing image: {str(e)}")

if __name__ == "__main__":
    main()

