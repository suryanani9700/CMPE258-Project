import streamlit as st
from PIL import Image
import torch
from transformers import CLIPModel, CLIPProcessor
import tempfile
from openai import OpenAI
from rare_diagnosis import *
import openai


api_key = "use your own openAI API Key "
client = OpenAI(api_key = api_key) 

# Define the inference function
def infer(image_path, texts, model_path="fine_tuned_clip_model.pt", device="cuda" if torch.cuda.is_available() else "cpu"):
    model = CLIPModel.from_pretrained(model_path).to(device)
    processor = CLIPProcessor.from_pretrained(model_path)
    
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, text=texts, return_tensors="pt", padding=True, truncation=True, max_length=77)
    
    pixel_values = inputs["pixel_values"].to(device)
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    
    model.eval()
    with torch.no_grad():
        outputs = model(pixel_values=pixel_values, input_ids=input_ids, attention_mask=attention_mask)
    
    logits_per_image = outputs.logits_per_image
    best_idx = torch.argmax(logits_per_image, dim=1).item()
    best_text = texts[best_idx]
    best_score = logits_per_image[0, best_idx].item()
    
    return best_text, best_score

# --------------------------------------------
# Streamlit App
st.title("Contrastive Learning for Chest X-Rays")
st.markdown("Upload a chest X-ray image, and we will find the most likely rare diagnosis description.")

uploaded_file = st.file_uploader("Upload a radiology image (JPEG/PNG):", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Radiology Image", use_column_width=True)

    # Perform inference when the user clicks the button
    if st.button("Generate Report"):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            tmp.write(uploaded_file.getvalue())
            tmp_path = tmp.name

        best_match, score = infer(tmp_path, rare_chest_diagnoses)
        st.write("Report: ")
        st.write(best_match)

        completion = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                {"role": "system", "content": "You are an expert in understanidng medical language and terms."},
                {
                    "role": "user",
                    "content":f"Write a simplified report in plain english using the following text {best_match}"
                }
                ]
                )
        st.write("Simplified report: ")
        st.write(completion.choices[0].message.content.strip())
else:
    st.info("Please upload a radiology image to get started.")
