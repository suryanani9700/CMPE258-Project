import nltk
nltk.download('punkt')
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from textstat import textstat
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
from PIL import Image
import torch


def use_baseline(image_path: str) -> str:
    # normal image captioning model
    model_name = "nlpconnect/vit-gpt2-image-captioning"
    model = VisionEncoderDecoderModel.from_pretrained(model_name)
    processor = ViTImageProcessor.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    image = Image.open(image_path).convert("RGB")
    pixel_values = processor(images=image, return_tensors="pt").pixel_values

    # Generate caption
    with torch.no_grad():
        output_ids = model.generate(pixel_values, max_length=50, num_beams=5)
    caption = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    return caption


def compute_scores(generated_text, reference_text):
    reference_tokens = [nltk.word_tokenize(reference_text.lower())]
    candidate_tokens = nltk.word_tokenize(generated_text.lower())
    
    smoothie = SmoothingFunction().method4
    bleu = sentence_bleu(reference_tokens, candidate_tokens, smoothing_function=smoothie)

    # Compute Flesch Reading Ease score
    readability = textstat.flesch_reading_ease(generated_text)

    return bleu,readability
    

#bleu_score, readability_score = compute_scores(generated_text, reference_text)
#use_baseline(image_path)
#print("BLEU Score:", bleu_score)
#print("Flesch Reading Ease:", readability_score)
