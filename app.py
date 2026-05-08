import streamlit as st
import torch
from PIL import Image
import io
import os
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration, BitsAndBytesConfig
from peft import PeftModel

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Markdown Generator - Qwen2-VL",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================
# CUSTOM STYLING
# =========================
st.markdown("""
<style>
    .main {
        padding-top: 2rem;
    }
    .stButton button {
        width: 100%;
        padding: 0.5rem;
        border-radius: 0.5rem;
    }
    .output-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)

MODEL_NAME = "Qwen/Qwen2-VL-2B-Instruct"
CHECKPOINT_PATH = "./qlora-vlm"
IMAGE_SIZE = 512

@st.cache_resource
def load_model():
    try:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
        
        processor = AutoProcessor.from_pretrained(MODEL_NAME)
        
        base_model = Qwen2VLForConditionalGeneration.from_pretrained(
            MODEL_NAME,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=True
        )
        
        # load lora adapters
        if os.path.exists(CHECKPOINT_PATH):
            model = PeftModel.from_pretrained(base_model, CHECKPOINT_PATH)
        else:
            model = base_model
        
        model.eval()
        return processor, model
    
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None

def generate_markdown(processor, model, image_path_or_pil):
    try:
        if isinstance(image_path_or_pil, str):
            image = Image.open(image_path_or_pil).convert("RGB")
        else:
            image = image_path_or_pil.convert("RGB")
        
        image = image.resize((IMAGE_SIZE, IMAGE_SIZE))
        
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": "Please describe this image in markdown format, including any text, equations, or structured content you see."}
                ],
            }
        ]
        
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=text, images=[image], return_tensors="pt")
        
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            output_ids = model.generate(**inputs, max_new_tokens=1024)
        
        generated_text = processor.decode(output_ids[0], skip_special_tokens=True)
        
        if "assistant" in generated_text:
            result = generated_text.split("assistant")[-1].strip()
        else:
            result = generated_text
        
        return result
    
    except Exception as e:
        return f"Error generating markdown: {str(e)}"

def main():
    # Header
    st.title("Markdown Generator")
    st.subheader("Generate Markdown from Images using Qwen2-VL")

    # Load model
    with st.spinner("Loading model... This may take a moment on first run"):
        processor, model = load_model()
    
    if processor is None or model is None:
        st.error("Failed to load model. Please check the model path and configuration.")
        return
    
    st.success("Model loaded successfully!")
    st.divider()
    
    tab1, tab2 = st.tabs(["Upload Image", "ℹAbout"])
    
    with tab1:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("Upload or Paste Image")
            uploaded_file = st.file_uploader(
                "Choose an image",
                type=["jpg", "jpeg", "png", "webp", "gif"],
                help="Upload an image to generate markdown description"
            )
        
        with col2:
            st.subheader("Quick Actions")
            if st.button("Use Sample Image", use_container_width=True):
                st.session_state.use_sample = True
        
        st.divider()
        
        # Process image
        if uploaded_file is not None or st.session_state.get('use_sample', False):
            if uploaded_file is not None:
                image = Image.open(uploaded_file)
                st.session_state.use_sample = False
            else:
                # Create a simple sample image if no file uploaded
                image = Image.new('RGB', (512, 512), color='lightblue')
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Input Image")
                st.image(image, use_column_width=True)
            
            with col2:
                st.subheader("Generated Markdown")
                
                if st.button("Generate Markdown", use_container_width=True, type="primary"):
                    with st.spinner("Generating markdown... This may take some time"):
                        markdown_result = generate_markdown(processor, model, image)
                    
                    st.session_state.markdown_result = markdown_result
            
            if st.session_state.get('markdown_result'):
                result_text = st.session_state.markdown_result
                
                st.markdown(result_text)
                
                with st.expander("View Raw Text"):
                    st.code(result_text, language="markdown")

if __name__ == "__main__":
    main()
