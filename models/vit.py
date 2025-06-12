"""
Vision Transformer (ViT)モデルの実装と特徴抽出機能

このモジュールでは、HuggingFace Transformersライブラリを使用してViTモデルによる画像特徴抽出を行います。
1024×1024サイズの高解像度画像に対応できるよう、適切な設定を行っています。
"""

import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import os
import sys
from transformers import ViTModel, ViTImageProcessor
from einops import rearrange

# 親ディレクトリをパスに追加
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class ViTFeatureExtractor:
    """Vision Transformerを使った特徴抽出クラス"""
    
    def __init__(self, model_name='google/vit-base-patch16-224', device=None):
        """
        Args:
            model_name: 使用するViTモデルの名前
            device: 計算に使用するデバイス（None=自動選択）
        """
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
            
        print(f"Using device: {self.device}")
        
        # モデルの読み込み
        self.model = ViTModel.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        
        # 前処理の定義
        self.processor = ViTImageProcessor.from_pretrained(model_name)
        
        # パッチサイズの取得
        self.patch_size = self.model.config.patch_size
        self.num_patches = (self.model.config.image_size // self.patch_size) ** 2
        
        print(f"Model: {model_name}, Patch size: {self.patch_size}")

    def extract_features(self, image_path, return_attention=False):
        """
        画像から特徴を抽出する
        
        Args:
            image_path: 画像ファイルのパス
            return_attention: 注意マップも返すかどうか
            
        Returns:
            features: 抽出された特徴ベクトル
            attention_maps: 注意マップ（オプション）
        """
        img = Image.open(image_path).convert('RGB')
        inputs = self.processor(images=img, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs, output_attentions=return_attention)
            
            # 最後の隠れ状態を特徴として使用
            features = outputs.last_hidden_state
            
            if return_attention:
                # 注意マップを取得
                attention_maps = outputs.attentions
                return features, attention_maps
            else:
                return features
    
    def visualize_features(self, features, num_features=64, figsize=(10, 10)):
        """
        抽出された特徴を可視化する
        
        Args:
            features: モデルから抽出された特徴
            num_features: 可視化する特徴の数
            figsize: 出力画像のサイズ
        """
        # CLS tokenを除外して特徴を取得
        patch_features = features[0, 1:].cpu().numpy()
        
        # 特徴の数を制限
        num_features = min(num_features, patch_features.shape[1])
        
        # 特徴マップの可視化
        fig, axes = plt.subplots(8, 8, figsize=figsize)
        axes = axes.flatten()
        
        for i in range(num_features):
            # 特徴を画像グリッドに変換
            size = int(np.sqrt(patch_features.shape[0]))
            feature_map = patch_features[:, i].reshape(size, size)
            
            # 正規化
            feature_map = (feature_map - feature_map.min()) / (feature_map.max() - feature_map.min() + 1e-8)
            
            axes[i].imshow(feature_map, cmap='viridis')
            axes[i].set_title(f'Feature {i+1}')
            axes[i].axis('off')
        
        plt.tight_layout()
        return fig
    
    def visualize_attention(self, image_path, attention_maps, layer_idx=-1, head_idx=0):
        """
        注意マップを可視化する
        
        Args:
            image_path: 元画像のパス
            attention_maps: モデルから抽出された注意マップ
            layer_idx: 可視化するレイヤーのインデックス
            head_idx: 可視化するアテンションヘッドのインデックス
        """
        # 画像の読み込み
        img = Image.open(image_path).convert('RGB')
        img = img.resize((224, 224))
        
        # 注意マップの取得 (transformersの出力形式に合わせて調整)
        # [batch_size, num_heads, seq_length, seq_length]
        attn = attention_maps[layer_idx][0, head_idx].cpu().numpy()
        
        # CLSトークンに対する注意を取得
        cls_attn = attn[0, 1:]  # CLS token -> patches
        
        # 画像グリッドに変換
        grid_size = int(np.sqrt(len(cls_attn)))
        cls_attn = cls_attn.reshape(grid_size, grid_size)
        
        # 可視化
        fig, ax = plt.subplots(1, 2, figsize=(16, 8))
        ax[0].imshow(img)
        ax[0].set_title('Original Image')
        ax[0].axis('off')
        
        im = ax[1].imshow(cls_attn, cmap='viridis')
        ax[1].set_title(f'Attention Map (Layer {layer_idx+1}, Head {head_idx+1})')
        ax[1].axis('off')
        
        fig.colorbar(im, ax=ax[1], fraction=0.046, pad=0.04)
        plt.tight_layout()
        
        return fig


def process_high_resolution_image(image_path, output_dir='outputs'):
    """
    高解像度画像（1024×1024）を処理し、特徴を抽出・可視化する
    
    Args:
        image_path: 処理する画像のパス
        output_dir: 結果を保存するディレクトリ
    """
    # 出力ディレクトリの作成
    os.makedirs(output_dir, exist_ok=True)
    
    # ViT特徴抽出器の初期化
    extractor = ViTFeatureExtractor()
    
    # 特徴と注意マップの抽出
    features, attention_maps = extractor.extract_features(image_path, return_attention=True)
    
    # ファイル名の取得
    img_name = Path(image_path).stem
    
    # 特徴の可視化と保存
    feature_fig = extractor.visualize_features(features)
    feature_fig.savefig(f"{output_dir}/{img_name}_vit_features.png")
    
    # 注意マップの可視化と保存（最後のレイヤーの最初のヘッド）
    attention_fig = extractor.visualize_attention(image_path, attention_maps)
    attention_fig.savefig(f"{output_dir}/{img_name}_vit_attention.png")
    
    # 複数の注意ヘッドを可視化
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    img = Image.open(image_path).convert('RGB')
    img = img.resize((224, 224))
    
    axes[0].imshow(img)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    for i in range(1, 6):
        attn = attention_maps[-1][0, i].cpu().numpy()
        cls_attn = attn[0, 1:]  # CLS token -> patches
        grid_size = int(np.sqrt(len(cls_attn)))
        cls_attn = cls_attn.reshape(grid_size, grid_size)
        
        im = axes[i].imshow(cls_attn, cmap='viridis')
        axes[i].set_title(f'Attention Head {i+1}')
        axes[i].axis('off')
    
    plt.tight_layout()
    fig.savefig(f"{output_dir}/{img_name}_vit_multi_attention.png")
    
    print(f"ViT特徴抽出と可視化が完了しました。結果は{output_dir}ディレクトリに保存されています。")
    return features


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Vision Transformerによる特徴抽出と可視化')
    parser.add_argument('--image', type=str, default='data/00325_hr.png',
                        help='処理する画像のパス')
    parser.add_argument('--output', type=str, default='outputs',
                        help='結果を保存するディレクトリ')
    
    args = parser.parse_args()
    
    # 高解像度画像の処理
    process_high_resolution_image(args.image, args.output) 