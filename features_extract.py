import net
import torch
import os
import numpy as np
import PIL
from PIL import Image
from collections import defaultdict

adaface_models = {
    'ir_18': "pretrained_adaface/adaface_ir18_casia.ckpt"
}

def remove_outliers(features, threshold=3):
    mean = torch.mean(features, dim=0)
    std = torch.std(features, dim=0)
    mask = torch.all(torch.abs(features - mean) < (threshold * std), dim=1)
    filtered_features = features[mask]
    return filtered_features

def load_pretrained_model(architecture='ir_18'):
    assert architecture in adaface_models.keys()
    model = net.build_model(architecture)
    statedict = torch.load(adaface_models[architecture])['state_dict']
    model_statedict = {key[6:]: val for key, val in statedict.items() if key.startswith('model.')}
    model.load_state_dict(model_statedict)
    model.eval()
    return model

def to_input(pil_rgb_image):
    np_img = np.array(pil_rgb_image)
    brg_img = ((np_img[:, :, ::-1] / 255.) - 0.5) / 0.5
    tensor = torch.tensor([brg_img.transpose(2, 0, 1)]).float()
    return tensor

path_features = './sample_imgs/features'
features = "ir_18"

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_pretrained_model(features).to(device)

    total_paths = [os.path.join(path_features, 'sample_features_celeba.t')]
    image_paths = ['./sample_imgs/LR_imgs']

    for total_path_features, image_path in zip(total_paths, image_paths):
        print(f"Processando diretório: {image_path}")
        print(f"Salvando features em: {total_path_features}")

        features_by_identity = defaultdict(list)

        j = 0
        for identity_dir in sorted(os.listdir(image_path)):
            identity_path = os.path.join(image_path, identity_dir)
            if not os.path.isdir(identity_path):
                continue  # Ignorar se não for diretório

            for img_name in sorted(os.listdir(identity_path)):
                img_path = os.path.join(identity_path, img_name)
                img = PIL.Image.open(img_path)
                aligned_rgb_img = img.resize((112, 112))
                bgr_tensor_input = to_input(aligned_rgb_img).to(device)

                with torch.no_grad():
                    feature_gl, _ = model(bgr_tensor_input)
                
                features_by_identity[identity_dir].append(feature_gl.cpu())

                j += 1
                if j % 10 == 0:
                    print(f"Processando imagem {j}: {img_name}, ID: {identity_dir}")

        mean_features_by_identity = {}

        for identity, features in features_by_identity.items():
            stacked_features = torch.stack(features)
            mean_feature = torch.mean(stacked_features, dim=0)
            mean_features_by_identity[identity] = mean_feature

        features_list = [mean_features_by_identity[identity] for identity in sorted(mean_features_by_identity.keys())]
        feat_pro = torch.stack(features_list)
        with open(total_path_features, 'wb') as f:
            torch.save(feat_pro, f)

        print(f"Features médias salvas em {total_path_features}")

