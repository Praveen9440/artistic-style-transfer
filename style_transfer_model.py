import torch
import torchvision
import torchvision.models as models
from torchvision import transforms
from PIL import Image
import numpy as np
import random

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)

# Set device (GPU if available, otherwise CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load pre-trained VGG19 model
vgg19 = models.vgg19(pretrained=True).features

# Freeze all parameters - we won't be training the model
for param in vgg19.parameters():
    param.requires_grad = False

# Move model to appropriate device
vgg19 = vgg19.to(device)

# Set model to evaluation mode
vgg19.eval()

print("VGG19 model loaded and frozen successfully")

class VGGFeatureExtractor(torch.nn.Module):
    def __init__(self, vgg_features):
        super(VGGFeatureExtractor, self).__init__()
        self.vgg_features = vgg_features
        
        # Define the layer indices for style and content layers
        self.style_layers = {
            'conv1_1': 0,
            'conv2_1': 5,
            'conv3_1': 10,
            'conv4_1': 19,
            'conv5_1': 28
        }
        
        self.content_layers = {
            'conv4_2': 21
        }
    
    def forward(self, x):
        features = {}
        
        # Extract features from each layer
        for i, layer in enumerate(self.vgg_features):
            x = layer(x)
            
            # Check if current layer is a style layer
            for layer_name, layer_idx in self.style_layers.items():
                if i == layer_idx:
                    features[layer_name] = x
            
            # Check if current layer is a content layer
            for layer_name, layer_idx in self.content_layers.items():
                if i == layer_idx:
                    features[layer_name] = x
        
        return features

def content_loss(target_features, content_features, layer_name):
    """
    Calculate the Mean Squared Error between content feature maps
    of the target image and the original content image.
    """
    return torch.nn.functional.mse_loss(target_features[layer_name], content_features[layer_name])

def gram_matrix(feature_map):
    """
    Compute the Gram matrix of a feature map tensor.
    The Gram matrix captures the correlations between different feature maps.
    """
    batch_size, channels, height, width = feature_map.size()
    
    # Reshape the feature map to (batch_size, channels, height*width)
    features = feature_map.view(batch_size, channels, height * width)
    
    # Compute the Gram matrix: G = F * F^T
    gram = torch.bmm(features, features.transpose(1, 2))
    
    # Normalize by the number of elements
    return gram / (channels * height * width)

def style_loss(target_features, style_features, style_layers):
    """
    Calculate the total style loss by summing the MSE between the Gram matrices
    of the target image and the style image across all style layers.
    """
    total_style_loss = 0
    
    for layer_name in style_layers:
        target_gram = gram_matrix(target_features[layer_name])
        style_gram = gram_matrix(style_features[layer_name])
        
        # Calculate MSE between Gram matrices
        layer_loss = torch.nn.functional.mse_loss(target_gram, style_gram)
        total_style_loss += layer_loss
    
    return total_style_loss

def load_image(image_path, max_size=512):
    """
    Load and preprocess an image for the VGG model.
    """
    image = Image.open(image_path).convert('RGB')
    
    # Resize if too large
    if max(image.size) > max_size:
        image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
    
    # Define transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Apply transforms and add batch dimension
    image_tensor = transform(image).unsqueeze(0)
    return image_tensor.to(device)

def save_image(tensor, output_path):
    """
    Convert tensor back to image and save it.
    """
    # Denormalize the tensor
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
    
    tensor = tensor * std + mean
    tensor = torch.clamp(tensor, 0, 1)
    
    # Convert to PIL Image
    image = transforms.ToPILImage()(tensor.squeeze(0).cpu())
    image.save(output_path)

def run_style_transfer(content_img_path, style_img_path, output_img_path, 
                      num_steps=300, style_weight=1000000, content_weight=1):
    """
    Main function to perform neural style transfer.
    """
    print("Loading images...")
    content_tensor = load_image(content_img_path)
    style_tensor = load_image(style_img_path)
    
    print("Creating VGG feature extractor...")
    feature_extractor = VGGFeatureExtractor(vgg19)
    
    print("Extracting content and style features...")
    with torch.no_grad():
        content_features = feature_extractor(content_tensor)
        style_features = feature_extractor(style_tensor)
    
    # Initialize target image as a clone of content image
    target_tensor = content_tensor.clone().requires_grad_(True)
    
    print("Setting up optimizer...")
    # Use Adam optimizer for precise step control
    optimizer = torch.optim.Adam([target_tensor], lr=0.01)
    
    # Set deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    print(f"Starting optimization for {num_steps} steps...")
    
    # Optimization loop with precise step control
    for step in range(num_steps):
        optimizer.zero_grad()
        
        # Extract features from current target image
        target_features = feature_extractor(target_tensor)
        
        # Calculate losses
        content_loss_value = content_loss(target_features, content_features, 'conv4_2')
        style_loss_value = style_loss(target_features, style_features, 
                                    ['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1'])
        
        # Total loss
        total_loss = content_weight * content_loss_value + style_weight * style_loss_value
        
        # Backward pass
        total_loss.backward()
        
        # Update parameters
        optimizer.step()
        
        # Print progress every 10 steps
        if step % 10 == 0:
            print(f"Step {step}, "
                  f"Content Loss: {content_loss_value.item():.4f}, "
                  f"Style Loss: {style_loss_value.item():.4f}, "
                  f"Total Loss: {total_loss.item():.4f}")
    
    print("Saving result...")
    save_image(target_tensor, output_img_path)
    print(f"Style transfer completed! Result saved to {output_img_path}")
