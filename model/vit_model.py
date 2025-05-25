import torch
import torch.nn as nn
import timm
from timm.models.vision_transformer import VisionTransformer

class CountryViT(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        """
        Initialize a Vision Transformer model for country classification
        
        Args:
            num_classes (int): Number of country classes to predict
            pretrained (bool): Whether to use pretrained weights
        """
        super(CountryViT, self).__init__()
        
        # Load pretrained ViT model
        self.vit = timm.create_model(
            'vit_base_patch16_224',  # Using ViT-Base with 16x16 patches
            pretrained=pretrained,
            num_classes=0  # Remove the default classification head
        )
        
        # Add custom classification head for countries
        self.classifier = nn.Sequential(
            nn.Linear(self.vit.embed_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, x):
        # Get features from ViT backbone
        features = self.vit.forward_features(x)
        
        # Get the [CLS] token features (first token)
        cls_token = features[:, 0]
        
        # Apply classification head
        logits = self.classifier(cls_token)
        
        return logits

def create_vit_model(num_classes, pretrained=True):
    """
    Factory function to create a ViT model for country classification
    
    Args:
        num_classes (int): Number of country classes to predict
        pretrained (bool): Whether to use pretrained weights
        
    Returns:
        CountryViT: Initialized Vision Transformer model
    """
    model = CountryViT(num_classes=num_classes, pretrained=pretrained)
    return model 