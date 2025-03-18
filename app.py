import streamlit as st
from PIL import Image
import os
from dotenv import load_dotenv
import io
from groq import Groq
import base64

# Load environment variables
load_dotenv()

# Configure API key
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    st.error("Please set GROQ_API_KEY in .env file")
    st.stop()

# Configure Groq
groq_client = Groq(api_key=GROQ_API_KEY)

# Define available models
LLAMA_MODELS = {
    "Llama 3.2 90B Vision": "llama-3.2-90b-vision-preview",
    "Llama 3.2 11B Vision": "llama-3.2-11b-vision-preview"
}

# Configure Streamlit page
st.set_page_config(
    page_title="Content Extractor",
    page_icon="📝",
    layout="wide"
)

# Add custom CSS
st.markdown("""
    <style>
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    .stButton>button {
        width: 100%;
    }
    .model-selector {
        margin-bottom: 20px;
    }
    </style>
""", unsafe_allow_html=True)

def process_image(image):
    """Process image to ensure RGB format and reasonable size"""
    if image.mode == 'RGBA':
        image = image.convert('RGB')
    
    max_size = 1600
    if max(image.size) > max_size:
        ratio = max_size / max(image.size)
        new_size = tuple(int(dim * ratio) for dim in image.size)
        image = image.resize(new_size, Image.Resampling.LANCZOS)
    
    return image

def encode_image_to_base64(image):
    """Convert PIL Image to base64 string"""
    try:
        processed_image = process_image(image)
        buffered = io.BytesIO()
        processed_image.save(buffered, format="JPEG", quality=95)
        return base64.b64encode(buffered.getvalue()).decode()
    except Exception as e:
        st.error(f"Error processing image: {str(e)}")
        return None

def get_llama_response(input_image, model_name):
    """Extract exact content from image"""
    try:
        base64_image = encode_image_to_base64(input_image)
        if not base64_image:
            return None
        
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Read and transcribe the exact text content from this image. Do not describe the image or add any additional context."
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    }
                ]
            }
        ]
        
        completion = groq_client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=0.1,
            max_tokens=1024,
            top_p=1,
            stream=False
        )
        
        return completion.choices[0].message.content
    except Exception as e:
        st.error(f"API Error: {str(e)}")
        return None

def load_image():
    """Image loader and processor"""
    uploaded_file = st.file_uploader(
        "Upload an image", 
        type=['png', 'jpg', 'jpeg'],
        help="Supported formats: PNG, JPG, JPEG"
    )
    if uploaded_file is not None:
        try:
            image_data = uploaded_file.read()
            image = Image.open(io.BytesIO(image_data))
            return image
        except Exception as e:
            st.error(f"Error loading image: {str(e)}")
            return None
    return None

def main():
    st.title("📝 Content Extractor")
    st.markdown("---")

    # Model selection in sidebar
    st.sidebar.header("Model Selection")
    selected_model_name = st.sidebar.selectbox(
        "Select Model",
        options=list(LLAMA_MODELS.keys()),
        index=0,
        help="Choose the model for extraction"
    )
    selected_model = LLAMA_MODELS[selected_model_name]
    
    # Create two columns
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### Upload Image")
        image = load_image()
        if image:
            processed_image = process_image(image)
            st.image(processed_image, caption="Uploaded Image", use_container_width=True)
            
            if st.button("Extract Content", type="primary"):
                with st.spinner("Extracting content..."):
                    try:
                        with col2:
                            st.markdown("### Extracted Content")
                            content = get_llama_response(processed_image, selected_model)
                            
                            if content:
                                st.text_area("Content", value=content, height=300)
                                st.download_button(
                                    label="Download Content",
                                    data=content,
                                    file_name="extracted_content.txt",
                                    mime="text/plain"
                                )
                            else:
                                st.error("Failed to extract content. Please try again.")
                    except Exception as e:
                        st.error(f"Error during extraction: {str(e)}")

if __name__ == "__main__":
    main()