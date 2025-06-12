"""
DETR (DEtection TRansformer)モデルの実装と特徴抽出機能

このモジュールでは、HuggingFace TransformersライブラリのDETRモデルを使用して画像から物体検出と特徴抽出を行います。
1024×1024サイズの高解像度画像に対応できるよう調整しています。
"""

import torch
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import os
import sys
import torchvision.transforms as T
from transformers import DetrForObjectDetection, DetrImageProcessor
import matplotlib.patches as patches
import torch.nn as nn

# 親ディレクトリをパスに追加
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# COCO データセットのクラス
COCO_CLASSES = [
    'N/A', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A',
    'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
    'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack',
    'umbrella', 'N/A', 'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
    'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'N/A', 'wine glass',
    'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
    'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
    'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table', 'N/A',
    'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
    'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A',
    'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

# カラーマップ（物体ごとに異なる色を割り当て）
COLORS = np.random.randint(0, 255, size=(len(COCO_CLASSES), 3)) / 255.0


class DETRFeatureExtractor:
    """DETRを使った特徴抽出と物体検出クラス"""
    
    def __init__(self, model_name='facebook/detr-resnet-50', confidence_threshold=0.3, device=None):
        """
        Args:
            model_name: 使用するDETRモデルの名前
            confidence_threshold: 検出の確信度閾値
            device: 計算に使用するデバイス（None=自動選択）
        """
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
            
        print(f"Using device: {self.device}")
        
        # モデルの読み込み
        self.model = DetrForObjectDetection.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        
        # プロセッサの読み込み
        self.processor = DetrImageProcessor.from_pretrained(model_name)
        
        self.confidence_threshold = confidence_threshold
        print(f"Model loaded with confidence threshold: {confidence_threshold}")
        
    def detect_objects(self, image_path):
        """
        画像から物体を検出する
        
        Args:
            image_path: 画像ファイルのパス
            
        Returns:
            boxes: 検出されたバウンディングボックス
            labels: 検出されたクラスラベル
            scores: 検出の確信度スコア
        """
        img = Image.open(image_path).convert('RGB')
        inputs = self.processor(images=img, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # トランスフォーマーの出力を処理
        target_sizes = torch.tensor([img.size[::-1]]).to(self.device)
        results = self.processor.post_process_object_detection(
            outputs, 
            target_sizes=target_sizes, 
            threshold=self.confidence_threshold
        )[0]
        
        boxes = results["boxes"].cpu().numpy()
        scores = results["scores"].cpu().numpy()
        labels = results["labels"].cpu().numpy()
        
        return boxes, labels, scores, img
    
    def extract_features(self, image_path):
        """
        画像から特徴マップを抽出する
        
        Args:
            image_path: 画像ファイルのパス
            
        Returns:
            features: 抽出された特徴マップ
        """
        img = Image.open(image_path).convert('RGB')
        inputs = self.processor(images=img, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            # モデルの推論を実行し、バックボーンの出力を取得
            outputs = self.model(**inputs, output_hidden_states=True)
            
            # バックボーンの出力を使用
            # DETR ResNet50バックボーンの最後の層の特徴を取得
            # これはハイドゥンステートの最後の要素として返される
            if hasattr(outputs, 'encoder_hidden_states') and outputs.encoder_hidden_states is not None:
                # Transformerのエンコーダ出力を使用
                features = outputs.encoder_hidden_states[-1]
            elif hasattr(outputs, 'hidden_states') and outputs.hidden_states is not None:
                # 通常のhidden_statesを使用
                features = outputs.hidden_states[-1]
            else:
                # 上記が利用できない場合はラストリゾートとして直接バックボーンにアクセス
                # 再度推論を行う
                with torch.no_grad():
                    features = []
                    
                    def hook_fn(module, input, output):
                        features.append(output)
                    
                    # バックボーンの最終層にフックを追加
                    if hasattr(self.model, 'model') and hasattr(self.model.model, 'backbone'):
                        hook = self.model.model.backbone.register_forward_hook(hook_fn)
                    else:
                        hook = self.model.backbone.register_forward_hook(hook_fn)
                    
                    # 推論を実行
                    _ = self.model(**inputs)
                    
                    # フックを削除
                    hook.remove()
                    
                    # 最後の特徴マップを使用
                    if features:
                        features = features[0]
                    else:
                        raise ValueError("特徴マップを抽出できませんでした。")
            
            return features
    
    def visualize_detection(self, image_path, boxes, labels, scores, figsize=(12, 10)):
        """
        検出結果を可視化する
        
        Args:
            image_path: 画像ファイルのパス
            boxes: 検出されたバウンディングボックス
            labels: 検出されたクラスラベル
            scores: 検出の確信度スコア
            figsize: 出力画像のサイズ
            
        Returns:
            fig: 可視化結果の図
        """
        img = Image.open(image_path).convert('RGB')
        
        fig, ax = plt.subplots(figsize=figsize)
        ax.imshow(img)
        
        for box, label, score in zip(boxes, labels, scores):
            # バウンディングボックスの描画
            x1, y1, x2, y2 = box
            rect = patches.Rectangle(
                (x1, y1), x2 - x1, y2 - y1, 
                linewidth=2, 
                edgecolor=COLORS[label], 
                facecolor='none'
            )
            ax.add_patch(rect)
            
            # ラベルテキストの描画
            class_name = COCO_CLASSES[label]
            text = f'{class_name} {score:.2f}'
            ax.text(
                x1, y1, text, 
                fontsize=10, 
                bbox=dict(facecolor=COLORS[label], alpha=0.5)
            )
        
        ax.set_title('DETR Object Detection')
        ax.axis('off')
        
        plt.tight_layout()
        return fig
    
    def visualize_features(self, features, figsize=(12, 12)):
        """
        抽出された特徴マップを可視化する
        
        Args:
            features: モデルから抽出された特徴マップ（テンソルまたはタプル）
            figsize: 出力画像のサイズ
            
        Returns:
            fig: 可視化結果の図
        """
        # 特徴マップがタプルの場合は最初の要素を取得
        if isinstance(features, tuple):
            features = features[0]
        
        # 特徴マップの形状を取得
        features_np = features.squeeze().cpu().numpy()
        
        # 特徴の次元数をチェック
        if len(features_np.shape) == 1:
            # 1次元の特徴ベクトルの場合（バーチャートとして表示）
            fig, ax = plt.subplots(figsize=(12, 6))
            
            # 表示する特徴数を制限
            num_features = min(256, len(features_np))
            feature_indices = np.arange(num_features)
            
            # 特徴値の正規化
            normalized_features = (features_np[:num_features] - features_np[:num_features].min()) / (features_np[:num_features].max() - features_np[:num_features].min() + 1e-8)
            
            # バーチャートを作成
            ax.bar(feature_indices, normalized_features)
            ax.set_xlabel('Feature Index')
            ax.set_ylabel('Normalized Feature Value')
            ax.set_title('DETR Feature Vector Visualization')
            plt.tight_layout()
            
            return fig
            
        elif len(features_np.shape) == 3:
            # 3次元特徴マップ [C, H, W] の場合
            num_channels = min(64, features_np.shape[0])
            grid_size = int(np.ceil(np.sqrt(num_channels)))
            
            fig, axes = plt.subplots(grid_size, grid_size, figsize=figsize)
            axes = axes.flatten()
            
            for i in range(num_channels):
                # 特徴マップの正規化
                feature_map = features_np[i]
                feature_map = (feature_map - feature_map.min()) / (feature_map.max() - feature_map.min() + 1e-8)
                
                axes[i].imshow(feature_map, cmap='viridis')
                axes[i].set_title(f'Channel {i+1}')
                axes[i].axis('off')
            
            # 使用しない軸を非表示
            for i in range(num_channels, len(axes)):
                axes[i].axis('off')
            
            plt.tight_layout()
            return fig
            
        else:
            # 2次元や4次元以上の場合は適切な形状に変換
            # 2次元の場合は1チャネルの画像として表示
            if len(features_np.shape) == 2:
                fig, ax = plt.subplots(figsize=(8, 8))
                feature_map = (features_np - features_np.min()) / (features_np.max() - features_np.min() + 1e-8)
                ax.imshow(feature_map, cmap='viridis')
                ax.set_title('DETR Feature Map')
                ax.axis('off')
                return fig
            # 4次元以上の場合は最初の次元だけを使用
            elif len(features_np.shape) >= 4:
                # [B, C, H, W] の場合、バッチの最初の要素を使用
                features_np = features_np[0]
                # 再帰的に呼び出し
                return self.visualize_features(torch.tensor(features_np), figsize)


def process_high_resolution_image_with_detr(image_path, output_dir='outputs'):
    """
    高解像度画像（1024×1024）をDETRで処理し、物体検出と特徴抽出を行う
    
    Args:
        image_path: 処理する画像のパス
        output_dir: 結果を保存するディレクトリ
        
    Returns:
        特徴マップ、バウンディングボックス、検出ラベル、検出スコアのタプル
    """
    # 出力ディレクトリの作成
    os.makedirs(output_dir, exist_ok=True)
    
    # DETR特徴抽出器の初期化
    extractor = DETRFeatureExtractor()
    
    # ファイル名の取得
    img_name = Path(image_path).stem
    
    # 物体検出
    boxes, labels, scores, img = extractor.detect_objects(image_path)
    
    # 検出結果の可視化と保存
    detection_fig = extractor.visualize_detection(image_path, boxes, labels, scores)
    detection_fig.savefig(f"{output_dir}/{img_name}_detr_detection.png")
    
    # 特徴マップの抽出
    features = extractor.extract_features(image_path)
    
    # 特徴マップの可視化と保存
    feature_fig = extractor.visualize_features(features)
    feature_fig.savefig(f"{output_dir}/{img_name}_detr_features.png")
    
    # 検出された物体の統計情報
    if len(labels) > 0:
        # 検出されたクラスの数をカウント
        unique_labels, counts = np.unique(labels, return_counts=True)
        class_names = [COCO_CLASSES[label] for label in unique_labels]
        
        # クラスごとの検出数を可視化
        plt.figure(figsize=(10, 6))
        plt.bar(class_names, counts)
        plt.xlabel('Object Classes')
        plt.ylabel('Count')
        plt.title('Detected Objects Distribution')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(f"{output_dir}/{img_name}_detr_class_distribution.png")
    
    print(f"DETR物体検出と特徴抽出が完了しました。結果は{output_dir}ディレクトリに保存されています。")
    print(f"検出された物体の数: {len(labels)}")
    
    # タプルではなく、4つの要素を個別に返す
    return features, boxes, labels, scores


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='DETRによる物体検出と特徴抽出')
    parser.add_argument('--image', type=str, default='data/00325_hr.png',
                        help='処理する画像のパス')
    parser.add_argument('--output', type=str, default='outputs',
                        help='結果を保存するディレクトリ')
    parser.add_argument('--threshold', type=float, default=0.7,
                        help='検出の確信度閾値')
    
    args = parser.parse_args()
    
    # 高解像度画像の処理
    process_high_resolution_image_with_detr(args.image, args.output) 