import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

#normalization using mean and standard deviation
clip_mean = [0.48145466, 0.4578275, 0.40821073]
clip_std = [0.26862954, 0.26130258, 0.27577711]

transform = transforms.Compose([
    transforms.GaussianBlur(kernel_size=(3, 3), sigma=(0.5, 1.0)),
    
    transforms.Resize((224, 224)),
    
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    
    transforms.ToTensor(),
    transforms.Normalize(mean=clip_mean, std=clip_std)
])

def denormalize(tensor, mean, std):
    tensor = tensor.clone() 
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return tensor

# Load and convert the image
image_path = "replace it with image path"  
image = Image.open(image_path).convert("RGB")

preprocessed_image = transform(image)

# Denormalize for visualization
denorm_img = denormalize(preprocessed_image, clip_mean, clip_std)
denorm_img = torch.clamp(denorm_img, 0, 1)

# Convert tensor from [C, H, W] to [H, W, C] for plotting
denorm_img_np = denorm_img.permute(1, 2, 0).numpy()

plt.imshow(denorm_img_np)
plt.axis('off')
plt.title("Preprocessed Image (Noise Reduced & Denormalized)")
plt.show()


""" References
[1] https://pytorch.org/vision/stable/transforms.html

[2] https://pytorch.org/tutorials/beginner/data_loading_tutorial.html

[3] https://pytorch.org/vision/0.11/transforms.html
"""