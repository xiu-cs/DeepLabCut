#
# DeepLabCut Toolbox (deeplabcut.org)
# © A. & M.W. Mathis Labs
# https://github.com/DeepLabCut/DeepLabCut
#
# Please see AUTHORS for contributors.
# https://github.com/DeepLabCut/DeepLabCut/blob/main/AUTHORS
#
# Licensed under GNU Lesser General Public License v3.0
#

import torch
import torch.nn as nn
import timm
from torch.hub import load_state_dict_from_url

from deeplabcut.pose_estimation_pytorch.models.backbones.base import (
    BACKBONES,
    BaseBackbone,
)


@BACKBONES.register_module
class ViT(BaseBackbone):
    """Vision Transformer (ViT) backbone with DINO pretraining support.
    
    This class implements a ViT backbone that can be used for pose estimation.
    It supports loading DINO pretrained weights for improved performance.
    """
    
    def __init__(
        self,
        model_name: str = "vit_base_patch16_224",
        img_size: int = 224,
        pretrained: bool = False,
        dino_pretrained: bool = True,
        dino_arch: str = "vit_base",
        patch_size: int = 16,
        drop_rate: float = 0.0,
        drop_path_rate: float = 0.1,
        **kwargs,
    ) -> None:
        """Initialize the ViT backbone.
        
        Args:
            model_name: Name of the ViT model architecture
            img_size: Input image size
            pretrained: If True, use ImageNet pretrained weights (ignored if dino_pretrained=True)
            dino_pretrained: If True, load DINO pretrained weights
            dino_arch: DINO architecture name (vit_small, vit_base, vit_large)
            patch_size: Patch size for the ViT model
            drop_rate: Dropout rate
            drop_path_rate: Stochastic depth drop-path rate
            **kwargs: Additional arguments passed to BaseBackbone
        """
        # ViT models typically have stride equal to patch size
        super().__init__(stride=patch_size, **kwargs)
        
        self.model_name = model_name
        self.img_size = img_size
        self.dino_pretrained = dino_pretrained
        self.dino_arch = dino_arch
        self.patch_size = patch_size
        
        if dino_pretrained:
            # Load DINO pretrained model
            self.model = self._load_dino_model(dino_arch, patch_size)
        elif pretrained:
            # Use timm model with optional ImageNet pretraining
            self.model = timm.create_model(
                model_name,
                pretrained=pretrained,
                img_size=(img_size, img_size),
                drop_rate=drop_rate,
                drop_path_rate=drop_path_rate,
                num_classes=0,  # Remove classification head
            )
        
        # Get the feature dimension
        self.feature_dim = self.model.embed_dim
        
    def _load_dino_model(self, arch: str, patch_size: int) -> nn.Module:
        """Load DINO pretrained ViT model.
        
        Args:
            arch: Architecture name (vit_small, vit_base, vit_large)
            patch_size: Patch size (8 or 16)
            
        Returns:
            DINO pretrained ViT model
        """
        # DINO model URLs (these are the official DINO checkpoints)
        dino_urls = {
            "vit_small_patch16": "https://dl.fbaipublicfiles.com/dino/dino_deitsmall16_pretrain/dino_deitsmall16_pretrain.pth",
            "vit_small_patch8": "https://dl.fbaipublicfiles.com/dino/dino_deitsmall8_pretrain/dino_deitsmall8_pretrain.pth",
            "vit_base_patch16": "https://dl.fbaipublicfiles.com/dino/dino_vitbase16_pretrain/dino_vitbase16_pretrain.pth",
            "vit_base_patch8": "https://dl.fbaipublicfiles.com/dino/dino_vitbase8_pretrain/dino_vitbase8_pretrain.pth",
        }
        
        # Map architecture names to timm model names
        arch_to_timm = {
            "vit_small": "vit_small_patch16_224" if patch_size == 16 else "vit_small_patch8_224",
            "vit_base": "vit_base_patch16_224" if patch_size == 16 else "vit_base_patch8_224",
        }
        
        if arch not in arch_to_timm:
            raise ValueError(f"Unsupported DINO architecture: {arch}")
            
        # Create model using timm
        model_name = arch_to_timm[arch]
        model = timm.create_model(
            model_name,
            pretrained=False,  # We'll load DINO weights manually
            img_size=(self.img_size, self.img_size),  # Ensure tuple format
            num_classes=0,  # Remove classification head
        )
        
        # Load DINO weights
        dino_key = f"{arch}_patch{patch_size}"
        if dino_key in dino_urls:
            print(f"Loading DINO pretrained weights for {dino_key}")
            try:
                state_dict = load_state_dict_from_url(dino_urls[dino_key], map_location="cpu")
                # DINO checkpoints have a 'teacher' key
                if 'teacher' in state_dict:
                    state_dict = state_dict['teacher']
                # Remove 'module.' prefix if present
                state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
                # Remove classification head weights if present
                state_dict = {k: v for k, v in state_dict.items() if not k.startswith('head.')}
                
                # Load state dict, handling positional embedding resize if needed
                try:
                    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
                    print(f"Loaded DINO weights with {len(missing_keys)} missing keys and {len(unexpected_keys)} unexpected keys")
                except RuntimeError as e:
                    if "pos_embed" in str(e):
                        # Handle positional embedding size mismatch
                        print("Handling positional embedding resize...")
                        # Get the positional embedding from checkpoint and model
                        checkpoint_pos_embed = state_dict['pos_embed']
                        model_pos_embed = model.pos_embed
                        
                        if checkpoint_pos_embed.shape != model_pos_embed.shape:
                            print(f"Resizing pos_embed from {checkpoint_pos_embed.shape} to {model_pos_embed.shape}")
                            # Remove pos_embed from state_dict and load the rest
                            state_dict_no_pos = {k: v for k, v in state_dict.items() if k != 'pos_embed'}
                            missing_keys, unexpected_keys = model.load_state_dict(state_dict_no_pos, strict=False)
                            print(f"Loaded DINO weights (without pos_embed) with {len(missing_keys)} missing keys")
                        else:
                            raise e
                    else:
                        raise e
                
            except Exception as e:
                print(f"Warning: Failed to load DINO weights: {e}")
                print("Falling back to ImageNet pretrained weights")
                model = timm.create_model(
                    model_name,
                    pretrained=True,
                    img_size=(self.img_size, self.img_size),
                    num_classes=0,
                )
        else:
            print(f"Warning: DINO weights not available for {dino_key}, using ImageNet weights")
            model = timm.create_model(
                model_name,
                pretrained=True,
                img_size=(self.img_size, self.img_size),
                num_classes=0,
            )
            
        return model
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the ViT backbone.
        
        Args:
            x: Input tensor of shape (batch_size, channels, height, width)
            
        Returns:
            Feature tensor of shape (batch_size, feature_dim, h', w')
            where h' and w' depend on the input size and patch size
        """
        # Get patch embeddings from ViT
        features = self.model.forward_features(x)
        
        # ViT outputs (batch_size, num_patches + 1, embed_dim)
        # Remove the class token (first token)
        if features.shape[1] > 1:
            features = features[:, 1:]  # Remove CLS token
        
        # Reshape to spatial format: (batch_size, embed_dim, h', w')
        batch_size, num_patches, embed_dim = features.shape
        h = w = int(num_patches ** 0.5)  # Assume square patches
        features = features.transpose(1, 2).reshape(batch_size, embed_dim, h, w)
        
        return features
    
    def get_feature_dim(self) -> int:
        """Get the output feature dimension."""
        return self.feature_dim