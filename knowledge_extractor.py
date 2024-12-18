from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np

# BioBert specializes on medical words
# ref: https://huggingface.co/dmis-lab/biobert-v1.1
tokenizer = AutoTokenizer.from_pretrained("dmis-lab/biobert-v1.1")
model = AutoModel.from_pretrained("dmis-lab/biobert-v1.1")

def get_bert_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def is_medical_word(word, similarity_threshold=0.7):
    # List of definitely medical words for comparison (related to chest x rays)
    medical_words = [
    "infiltrate",
    "consolidation",
    "nodule",
    "pleural",
    "effusion",
    "pneumothorax",
    "atelectasis",
    "fibrosis",
    "lymphadenopathy",
    "interstitial",
    "bronchiectasis",
    "emphysema",
    "cardiomegaly",
    "collapse",
    "calcification",
    "pulmonary",
    "alveolar",
    "ground-glass",
    "hyperinflation",
    "scarring",
    "opacity",
    "lesion",
    "interlobular",
    "septa",
    "hilar",
    "enlarged",
    "mass",
    "shadowing",
    "cavity",
    "perihilar",
    "reticular",
    "nodular",
    "sclerotic",
    "translucency",
    "bronchovascular",
    "lucency",
    "congestion",
    "mediastinal",
    "hyperlucent",
    "silhouette",
    "peribronchial",
    "honeycombing",
    "reticulonodular",
    "cavitation",
    "hemidiaphragm",
    "pneumonia",
    "granuloma",
    "edema",
    "airspace",
    "atelectatic"
  ]

    # Get embedding for the input word
    word_embedding = get_bert_embedding(word)
    
    # Compare with medical words
    # Note that we assume that the medical image would have similar similarity
    for medical_word in medical_words:
        medical_embedding = get_bert_embedding(medical_word)
        similarity = cosine_similarity(word_embedding, medical_embedding)
        
        if similarity > similarity_threshold:
            return True
    
    return False

def check_medical_term(word, similarity_threshold=0.7):
    result = is_medical_word(word, similarity_threshold)
    if result:
        return True
        #print(f"'{word}' is likely a medical term. (Threshold: {similarity_threshold})")
    else:
        return False
        #print(f"'{word}' is likely not a medical term. (Threshold: {similarity_threshold})")

# Example: check_medical_term("antibiotic", 0.8)
