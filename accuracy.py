import net
import torch
import os
import numpy as np
from PIL import Image
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity

# Predefined models and paths
adaface_models = {
    'ir_18': "pretrained_adaface/adaface_ir18_casia.ckpt"
}

def load_pretrained_model(architecture='ir_18'):
    """Load the pre-trained Adaface model."""
    assert architecture in adaface_models.keys()
    model = net.build_model(architecture)
    statedict = torch.load(adaface_models[architecture])['state_dict']
    model_statedict = {key[6:]: val for key, val in statedict.items() if key.startswith('model.')}
    model.load_state_dict(model_statedict)
    model.eval()
    return model

def to_input(pil_rgb_image):
    """Convert a PIL RGB image into a PyTorch tensor."""
    np_img = np.array(pil_rgb_image)
    bgr_img = ((np_img[:, :, ::-1] / 255.) - 0.5) / 0.5
    tensor = torch.tensor([bgr_img.transpose(2, 0, 1)]).float()
    return tensor

def extract_features(image_path, model, device):
    """Extract features from an image using the pre-trained model."""
    img = Image.open(image_path).resize((112, 112))
    tensor_input = to_input(img).to(device)

    with torch.no_grad():
        feature, _ = model(tensor_input)

    return feature.cpu().numpy()

def calculate_accuracy(features_gallery, features_sr):
    """Calculate accuracy by comparing gallery and super-resolution features."""
    correct = 0
    total = len(features_sr)

    # Compute cosine similarity between corresponding features
    similarities = cosine_similarity(features_sr, features_gallery)

    # For each SR image, find the corresponding gallery image with highest similarity
    predicted_indices = np.argmax(similarities, axis=1)

    # Check how many matches are correct
    for i, predicted in enumerate(predicted_indices):
        if i == predicted:
            correct += 1

    accuracy = correct / total
    return accuracy

def is_sr_image(filename):
    """Check if the file is a super-resolution image (suffix '-sr.png')."""
    return filename.endswith('-sr.png')

if __name__ == '__main__':
    # Device configuration and model loading
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_pretrained_model().to(device)

    # Image directories
    gallery_path = './sample_imgs/gallery'
    sr_results_path = './exps/sr_results'

    # Extract features from gallery images
    gallery_features = []
    for img_name in sorted(os.listdir(gallery_path)):
        img_path = os.path.join(gallery_path, img_name)
        feature = extract_features(img_path, model, device)
        gallery_features.append(feature)

    # Extract features from super-resolution images (only '-sr.png')
    sr_features = []
    sr_image_names = [f for f in sorted(os.listdir(sr_results_path)) if is_sr_image(f)]

    for img_name in sr_image_names:
        img_path = os.path.join(sr_results_path, img_name)
        feature = extract_features(img_path, model, device)
        sr_features.append(feature)

    # Convert feature lists to numpy arrays
    gallery_features = np.vstack(gallery_features)
    sr_features = np.vstack(sr_features)

    # Calculate accuracy
    accuracy = calculate_accuracy(gallery_features, sr_features)
    print(f"Facial recognition accuracy: {accuracy * 100:.2f}%")

