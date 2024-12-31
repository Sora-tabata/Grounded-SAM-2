import torch
import clip
from PIL import Image
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import json
import os
from collections import defaultdict

class PCAAnalyzer:
    def __init__(self, model_path, device):
        """
        Initialize the PCA analyzer with a trained CLIP model
        """
        self.device = device
        # Load the trained model state
        self.model, self.preprocess = clip.load("ViT-B/32", device=device, jit=False)
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
        
    def extract_features(self, image_paths, labels):
        """
        Extract features from images using the CLIP model
        """
        features_dict = defaultdict(list)
        
        with torch.no_grad():
            for img_path, label in tqdm(zip(image_paths, labels), desc="Extracting features"):
                # Load and preprocess image
                image = Image.open(img_path).convert("RGB")
                image = self.preprocess(image).unsqueeze(0).to(self.device)
                
                # Extract image features
                image_features = self.model.encode_image(image)
                features_dict[label].append(image_features.cpu().numpy().squeeze())
                
        return features_dict

    def perform_pca(self, features_dict, n_components=3):
        """
        Perform PCA on the extracted features
        """
        # Combine all features
        all_features = np.vstack([feat for feats in features_dict.values() for feat in feats])
        
        # Fit PCA
        pca = PCA(n_components=n_components)
        pca_features = pca.fit_transform(all_features)
        
        # Separate features by class
        start_idx = 0
        pca_by_class = {}
        for label, features in features_dict.items():
            end_idx = start_idx + len(features)
            pca_by_class[label] = pca_features[start_idx:end_idx]
            start_idx = end_idx
            
        return pca_by_class, pca.explained_variance_ratio_

    def visualize_pca(self, pca_by_class, variance_ratio, output_path):
        """
        Create visualizations of the PCA results
        """
        # 2D Scatter Plot
        plt.figure(figsize=(12, 8))
        for label, features in pca_by_class.items():
            plt.scatter(features[:, 0], features[:, 1], label=label, alpha=0.6)
        
        plt.xlabel(f'PC1 ({variance_ratio[0]:.2%} variance)')
        plt.ylabel(f'PC2 ({variance_ratio[1]:.2%} variance)')
        plt.title('PCA Analysis of CLIP Image Features by Class')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(os.path.join(output_path, 'pca_2d_scatter.png'))
        plt.close()

        # 3D Scatter Plot
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        for label, features in pca_by_class.items():
            ax.scatter(features[:, 0], features[:, 1], features[:, 2], label=label, alpha=0.6)
        
        ax.set_xlabel(f'PC1 ({variance_ratio[0]:.2%})')
        ax.set_ylabel(f'PC2 ({variance_ratio[1]:.2%})')
        ax.set_zlabel(f'PC3 ({variance_ratio[2]:.2%})')
        plt.title('3D PCA Analysis of CLIP Image Features')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(os.path.join(output_path, 'pca_3d_scatter.png'))
        plt.close()

def main():
    # Configuration
    CFG = {
        'device': torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        'model_path': '/mnt/source/Downloads/best_model_total.pt',
        'images_dir': '/mnt/source/cityscapes/train',
        'labels_json': '/mnt/source/cityscapes/annotations/cityscapes+panoptic/labels.json',
        'output_dir': '/mnt/source/datasets/TOTAL_CLIP/pca_analysis',
        'n_components': 3
    }
    
    # Create output directory if it doesn't exist
    os.makedirs(CFG['output_dir'], exist_ok=True)
    
    # Load image paths and labels
    with open(CFG['labels_json'], 'r', encoding='utf-8') as f:
        labels_data = json.load(f)
    
    image_paths = []
    labels = []
    for entry in labels_data:
        image_path = os.path.join(CFG['images_dir'], entry['image_name'])
        if os.path.isfile(image_path):
            image_paths.append(image_path)
            labels.append(entry['label'])
    
    # Initialize PCA analyzer
    analyzer = PCAAnalyzer(CFG['model_path'], CFG['device'])
    
    # Extract features
    features_dict = analyzer.extract_features(image_paths, labels)
    
    # Perform PCA
    pca_by_class, variance_ratio = analyzer.perform_pca(features_dict, CFG['n_components'])
    
    # Visualize results
    analyzer.visualize_pca(pca_by_class, variance_ratio, CFG['output_dir'])
    
    # Print variance explained
    print("\nVariance explained by principal components:")
    for i, var in enumerate(variance_ratio):
        print(f"PC{i+1}: {var:.2%}")

if __name__ == "__main__":
    main()