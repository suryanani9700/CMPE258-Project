import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor
from rare_diagnosis import *

import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

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
    
    # Find the index of the text with the highest similarity score
    best_idx = torch.argmax(logits_per_image, dim=1).item()
    best_text = texts[best_idx]
    best_score = logits_per_image[0, best_idx].item()
    
    return best_text, best_score

# Example usage:
#best_match, score = infer("/Users/sasankperumal/Documents/258/images/test_image.jpg", rare_chest_diagnoses)
#print("Best match text:", best_match)
#print("Similarity score:", score)