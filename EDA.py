import os
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from collections import Counter
from nltk.corpus import stopwords
import nltk

common_words = [
    "multiple",
    "seen",
    "rib",
    "aspiration",
    "lungs",
    "none",
    "r",
    "f",
    "heart",
    "size",
    "provided",
    "base",
    "fractures",
    "left ",
    "left",
    "normal",
    "lateral",
    "final",
    "report",
    "examination",
    "chest",
    "pa",
    "lat",
    "indication",
    "ascites",
    "infection",
    "technique",
    "comparison",
    "findings",
    "focal",
    "consolidation",
    "pleural",
    "effusion",
    "pneumothorax",
    "bilateral",
    "nodular",
    "opacities",
    "nipple",
    "shadows",
    "cardiomediastinal",
    "silhouette",
    "clips",
    "lung",
    "breast",
    "unremarkable",
    "chronic",
    "deformity",
    "posterior",
    "ribs",
    "impression",
    "acute",
    "cardiopulmonary",
    "process"
]


nltk.download('stopwords')

def load_patient_reports(root_folder):
    patient_data = {}
    for root, dirs, files in os.walk(root_folder):
        for file in files:
            if file.endswith(".txt"):
                patient_id = root.split("/")[-1] 
                with open(os.path.join(root, file), 'r') as f:
                    content = f.read()
                    if patient_id not in patient_data:
                        patient_data[patient_id] = []
                    patient_data[patient_id].append(content)
    return patient_data

def preprocess_text(reports):
    combined_text = " ".join(reports)
    text = re.sub(r'[^a-zA-Z\s]', '', combined_text)  # Remove special characters
    text = ' '.join(text.split())  # Normalize spaces
    return text.lower()

def avg_characters_per_patient(patient_reports):
    avg_chars = {patient: len(report) for patient, report in patient_reports.items()}
    overall_avg = np.mean(list(avg_chars.values()))
    print(f"Average number of characters per patient: {overall_avg:.2f}")
    return avg_chars

def most_common_keywords(patient_reports, top_n=5):
    stop_words = set(stopwords.words('english'))
    all_text = " ".join(patient_reports.values())
    words = all_text.split()
    words = [word for word in words if word not in stop_words]
    words = [word for word in words if word not in common_words]
    word_freq = Counter(words)
    common_keywords = word_freq.most_common(top_n)
    print("Most Common Keywords:")
    for word, freq in common_keywords:
        print(f"{word}: {freq} occurrences")
    return common_keywords

def calculate_similarity(patient_reports):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(patient_reports.values())
    similarity_matrix = cosine_similarity(tfidf_matrix)
    avg_similarity = np.mean(similarity_matrix[np.triu_indices_from(similarity_matrix, k=1)])
    print(f"Average Cosine Similarity between patients: {avg_similarity:.2f}")
    return similarity_matrix, avg_similarity

def plot_keywords(common_keywords):
    words, freqs = zip(*common_keywords)
    plt.figure(figsize=(8, 5))
    plt.bar(words, freqs)
    plt.xlabel('Keywords')
    plt.ylabel('Frequency')
    plt.title('Top Keywords in Radiology Reports')
    plt.show()

if __name__ == "__main__":
    root_folder = "folder here"  

    patient_data = load_patient_reports(root_folder)
    
    patient_reports = {patient: preprocess_text(reports) for patient, reports in patient_data.items()}
    
    avg_chars = avg_characters_per_patient(patient_reports)

    common_keywords = most_common_keywords(patient_reports, top_n=5)
    
    similarity_matrix, avg_similarity = calculate_similarity(patient_reports)

    plot_keywords(common_keywords)



"""References
[1] https://www.geeksforgeeks.org/text-preprocessing-for-nlp-tasks/ 

[2] https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html

[3] https://matplotlib.org/stable/tutorials/introductory/pyplot.html
"""