import torch
import os
import timm
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define image size and transformation process
image_size = 224
transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

# Function to run the model on the input image
def run(img_name, image):
    os.chdir(f'C:/Users/dlrkd/Desktop/graduation_work/posture_classfication')
    # Load the pre-trained model
    model = timm.create_model('vit_tiny_patch16_224', pretrained=False)  
    model.load_state_dict(torch.load(f'./model_weight_vit_tiny_patch16_224.pth',weights_only=False))
    model.to(device)
    model.eval()

    class_labels = {0: 'supine', 1: 'left', 2: 'right', 3: 'prone'}
    
    # Preprocess the image
    image = image.convert('RGB')
    image = transform(image).unsqueeze(0).to(device)

    # Make prediction with the model
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
        predicted_class = predicted.item()

        # Get the class name from the class labels dictionary
        class_name = class_labels.get(predicted_class)
        print(f'Image: {img_name} -> Predicted Class: {class_name} (Class Index: {predicted_class})')
    return 