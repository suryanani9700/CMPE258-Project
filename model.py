import os
from PIL import Image
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import CLIPProcessor, CLIPModel

image_path = "/Users/sasankperumal/Documents/258/images/test_image.jpg" 
report_path = "/Users/sasankperumal/Documents/258/s50414267.txt" 
device = "cuda" if torch.cuda.is_available() else "cpu"

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
model = model.to(device)


def extract_description(report_path):
    findings = []
    with open(report_path, 'r') as f:
        is_findings_section = False
        for line in f:
            line = line.strip()
            if line.startswith("FINDINGS:"):
                is_findings_section = True
                continue
            if line.startswith("IMPRESSION:"):
                break
            if is_findings_section:
                findings.append(line)
    return " ".join(findings).strip()

description = extract_description(report_path)
print(f"Extracted Description: {description}")


class SingleImageTextDataset(Dataset):
    def __init__(self, image_path, description, processor):
        self.image_path = image_path
        self.description = description
        self.processor = processor

    def __len__(self):
        return 1 

    def __getitem__(self, idx):
        image = Image.open(self.image_path).convert("RGB")

        # truncating the model to max_length as the model we are using has a toxen max length of 77
        processed = self.processor(
            images=image,
            text=self.description,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=77   
        )

        pixel_values = processed["pixel_values"].squeeze(0)  
        input_ids = processed["input_ids"].squeeze(0)  
        attention_mask = processed["attention_mask"].squeeze(0) 

        return pixel_values, input_ids, attention_mask

# training the model 
batch_size = 1
num_epochs = 30
learning_rate = 5e-5

dataset = SingleImageTextDataset(image_path, description, processor)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, betas=(0.9, 0.98), eps=1e-6, weight_decay=0.2)
loss_img = nn.CrossEntropyLoss()
loss_txt = nn.CrossEntropyLoss()


for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0
    pbar = tqdm(dataloader, total=len(dataloader), desc=f"Epoch {epoch+1}/{num_epochs}")

    for batch in pbar:
        pixel_values, input_ids, attention_mask = batch
        pixel_values = pixel_values.to(device)
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)

        optimizer.zero_grad()

        outputs = model(pixel_values=pixel_values, input_ids=input_ids, attention_mask=attention_mask)
        logits_per_image = outputs.logits_per_image
        logits_per_text = outputs.logits_per_text

        # compute the loss
        ground_truth = torch.arange(len(pixel_values), dtype=torch.long, device=device)
        total_loss = (loss_img(logits_per_image, ground_truth) + loss_txt(logits_per_text, ground_truth)) / 2

        total_loss.backward()
        optimizer.step()

        pbar.set_postfix(loss=total_loss.item())

    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss.item():.4f}")

# saving the fine model asd fine_tuned_clip_model.pt which consisits of both json and txt files
save_path = "fine_tuned_clip_model.pt"
os.makedirs(save_path, exist_ok=True)
model.save_pretrained(save_path)
processor.save_pretrained(save_path)

print(f"Model fine-tuning complete. Model saved to {save_path}.")





""" References

[1] https://huggingface.co/docs/transformers/training

[2] https://github.com/Zasder3/train-CLIP

[3] https://huggingface.co/blog/fine-tune-clip-rsicd

"""