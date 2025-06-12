"""
CLIP (Contrastive Language-Image Pre-training)モデルの実装と特徴抽出機能

このモジュールでは、OpenAIのCLIPモデルを使用して画像とテキストの特徴を抽出する機能を提供します。
1024×1024サイズの高解像度画像に対応できるよう調整しています。
"""

import torch
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import os
import sys
import clip
from sklearn.decomposition import PCA
import torch.nn.functional as F

# 親ディレクトリをパスに追加
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class CLIPFeatureExtractor:
    """CLIPを使った特徴抽出クラス"""
    
    def __init__(self, model_name='ViT-B/32', device=None):
        """
        Args:
            model_name: 使用するCLIPモデルの名前
            device: 計算に使用するデバイス（None=自動選択）
        """
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
            
        print(f"Using device: {self.device}")
        
        # モデルと前処理の読み込み
        self.model, self.preprocess = clip.load(model_name, device=self.device)
        
        print(f"Model: {model_name}")
        
    def extract_image_features(self, image_path):
        """
        画像から特徴を抽出する
        
        Args:
            image_path: 画像ファイルのパス
            
        Returns:
            features: 抽出された特徴ベクトル
        """
        img = Image.open(image_path).convert('RGB')
        img_tensor = self.preprocess(img).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            image_features = self.model.encode_image(img_tensor)
            # 正規化
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            
        return image_features
    
    def extract_text_features(self, texts):
        """
        テキストから特徴を抽出する
        
        Args:
            texts: テキストのリスト
            
        Returns:
            features: 抽出された特徴ベクトル
        """
        text_tokens = clip.tokenize(texts).to(self.device)
        
        with torch.no_grad():
            text_features = self.model.encode_text(text_tokens)
            # 正規化
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            
        return text_features
    
    def compare_image_with_texts(self, image_path, texts):
        """
        画像と複数のテキストの類似度を計算する
        
        Args:
            image_path: 画像ファイルのパス
            texts: テキストのリスト
            
        Returns:
            similarities: 類似度スコア
        """
        image_features = self.extract_image_features(image_path)
        text_features = self.extract_text_features(texts)
        
        # コサイン類似度を計算
        similarities = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        
        return similarities.cpu().numpy()[0]
    
    def visualize_similarities(self, image_path, texts, similarities, figsize=(12, 6)):
        """
        画像とテキストの類似度を可視化する
        
        Args:
            image_path: 画像ファイルのパス
            texts: テキストのリスト
            similarities: 類似度スコア
            figsize: 出力画像のサイズ
        """
        img = Image.open(image_path).convert('RGB')
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # 画像の表示
        ax1.imshow(img)
        ax1.set_title('Input Image')
        ax1.axis('off')
        
        # 類似度の表示
        y_pos = np.arange(len(texts))
        ax2.barh(y_pos, similarities * 100)
        ax2.set_yticks(y_pos)
        ax2.set_yticklabels(texts)
        ax2.set_xlabel('Similarity (%)')
        ax2.set_title('CLIP Image-Text Similarity')
        ax2.set_xlim(0, 100)
        
        plt.tight_layout()
        return fig
    
    def visualize_image_features(self, image_features, figsize=(10, 6)):
        """
        抽出された画像特徴を可視化する
        
        Args:
            image_features: モデルから抽出された特徴
            figsize: 出力画像のサイズ
        """
        # 特徴ベクトルをCPUに移動
        features_np = image_features.cpu().numpy()[0]
        
        # ヒストグラムで特徴の分布を表示
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        ax1.hist(features_np, bins=50)
        ax1.set_title('CLIP Feature Distribution')
        ax1.set_xlabel('Feature Value')
        ax1.set_ylabel('Frequency')
        
        # 特徴の強さを表示
        feature_strength = np.abs(features_np)
        top_indices = np.argsort(feature_strength)[-20:]  # 上位20個の特徴
        
        ax2.barh(np.arange(len(top_indices)), feature_strength[top_indices])
        ax2.set_yticks(np.arange(len(top_indices)))
        ax2.set_yticklabels([f'F{i}' for i in top_indices])
        ax2.set_xlabel('Feature Strength')
        ax2.set_title('Top 20 Strongest Features')
        
        plt.tight_layout()
        return fig


def process_high_resolution_image_with_clip(image_path, output_dir='outputs'):
    """
    高解像度画像（1024×1024）をCLIPで処理し、特徴を抽出・可視化する
    
    Args:
        image_path: 処理する画像のパス
        output_dir: 結果を保存するディレクトリ
    """
    # 出力ディレクトリの作成
    os.makedirs(output_dir, exist_ok=True)
    
    # CLIP特徴抽出器の初期化
    extractor = CLIPFeatureExtractor()
    
    # 画像特徴の抽出
    image_features = extractor.extract_image_features(image_path)
    
    # ファイル名の取得
    img_name = Path(image_path).stem
    
    # 画像特徴の可視化と保存
    feature_fig = extractor.visualize_image_features(image_features)
    feature_fig.savefig(f"{output_dir}/{img_name}_clip_features.png")
    
    # いくつかのテキスト記述子を定義
    descriptions = [
        "自然の風景",
        "都市の景観",
        "人物の顔",
        "建物や建築物",
        "抽象的なアート",
        "動物や生き物",
        "食べ物や料理",
        "テクノロジーや機械"
    ]
    
    # 画像とテキストの類似度を計算
    similarities = extractor.compare_image_with_texts(image_path, descriptions)
    
    # 類似度の可視化と保存
    similarity_fig = extractor.visualize_similarities(image_path, descriptions, similarities)
    similarity_fig.savefig(f"{output_dir}/{img_name}_clip_similarities.png")
    
    # テキスト特徴との類似度に基づくゼロショット分類
    categories = [
        "風景写真", "ポートレート", "建築写真", "抽象画", 
        "動物写真", "夜景", "水中写真", "料理写真"
    ]
    
    # カテゴリとの類似度を計算
    category_similarities = extractor.compare_image_with_texts(image_path, categories)
    
    # カテゴリ類似度の可視化と保存
    category_fig = extractor.visualize_similarities(image_path, categories, category_similarities)
    category_fig.savefig(f"{output_dir}/{img_name}_clip_categories.png")
    
    print(f"CLIP特徴抽出と可視化が完了しました。結果は{output_dir}ディレクトリに保存されています。")
    return image_features


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='CLIPによる特徴抽出と可視化')
    parser.add_argument('--image', type=str, default='data/00325_hr.png',
                        help='処理する画像のパス')
    parser.add_argument('--output', type=str, default='outputs',
                        help='結果を保存するディレクトリ')
    
    args = parser.parse_args()
    
    # 高解像度画像の処理
    process_high_resolution_image_with_clip(args.image, args.output) 