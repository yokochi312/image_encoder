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
import cv2

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

    def extract_features(self, image_path, return_attention=False, return_hidden_states=False):
        """
        画像から特徴を抽出する
        
        Args:
            image_path: 画像ファイルのパス
            return_attention: 注意マップも返すかどうか
            return_hidden_states: 中間層の特徴も返すかどうか
            
        Returns:
            features: 抽出された特徴ベクトル
            attention_maps: 注意マップ（オプション）
            hidden_states: 中間層の特徴（オプション）
        """
        img = Image.open(image_path).convert('RGB')
        # モデルの入力用にリサイズ
        img_resized = img.resize((224, 224))
        inputs = self.processor(images=img_resized, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs, output_attentions=return_attention, output_hidden_states=return_hidden_states)
            
            # 最後の隠れ状態を特徴として使用
            features = outputs.last_hidden_state
            
            if return_attention and return_hidden_states:
                return features, outputs.attentions, outputs.hidden_states
            elif return_attention:
                return features, outputs.attentions
            elif return_hidden_states:
                return features, outputs.hidden_states
            else:
                return features
    
    def visualize_features(self, features, num_features=64, figsize=(10, 10), ax=None):
        """
        抽出された特徴を可視化する
        
        Args:
            features: モデルから抽出された特徴
            num_features: 可視化する特徴の数
            figsize: 出力画像のサイズ
            ax: 描画用のAxesオブジェクト（None=新規作成）
        """
        # 独立した図として可視化する場合
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
            ax.axis('off')
            self._visualize_features_grid(features, ax, num_features=num_features)
            plt.tight_layout()
            return fig
        else:
            # 既存のAxesに可視化する場合
            self._visualize_features_grid(features, ax, num_features=num_features)
            return None
    
    def _visualize_features_grid(self, features, ax, num_features=64, cmap='viridis'):
        """
        特徴を8x8グリッドで可視化する（内部メソッド）
        
        Args:
            features: モデルから抽出された特徴
            ax: 描画用のAxesオブジェクト
            num_features: 可視化する特徴の数
            cmap: カラーマップ
        """
        # CLS tokenを除外して特徴を取得
        patch_features = features[0, 1:].cpu().numpy()
        
        # 特徴の数を制限
        num_features = min(num_features, patch_features.shape[1])
        
        # グリッドのサイズを設定
        n_cols = 8
        n_rows = 8
        
        # グリッドを作成（余白を含む）
        cell_size = 16  # セルサイズを大きく
        grid_width = n_cols * cell_size + (n_cols - 1)  # 列間の境界線
        grid_height = n_rows * cell_size + (n_rows - 1)  # 行間の境界線
        grid = np.ones((grid_height, grid_width)) * 0.9  # 薄いグレーの背景
        
        # 各特徴を配置
        for i in range(min(num_features, n_rows * n_cols)):
            row = i // n_cols
            col = i % n_cols
            
            # グリッド内の位置を計算
            y_start = row * (cell_size + 1)
            x_start = col * (cell_size + 1)
            
            # 特徴を画像グリッドに変換
            size = int(np.sqrt(patch_features.shape[0]))
            feature_map = patch_features[:, i].reshape(size, size)
            
            # 正規化（より明確なコントラストのために）
            feature_map = (feature_map - feature_map.min()) / (feature_map.max() - feature_map.min() + 1e-8)
            
            # リサイズして配置
            feature_resized = cv2.resize(feature_map, (cell_size - 2, cell_size - 2))
            
            # 特徴マップを配置（上部に1ピクセル余白）
            grid[y_start+1:y_start+cell_size-1, x_start+1:x_start+cell_size-1] = feature_resized
        
        # グリッドの表示
        im = ax.imshow(grid, cmap=cmap)
        
        # 特徴番号の追加
        for i in range(min(num_features, n_rows * n_cols)):
            row = i // n_cols
            col = i % n_cols
            
            # グリッド内の位置を計算
            y_pos = row * (cell_size + 1) + cell_size - 4
            x_pos = col * (cell_size + 1) + cell_size // 2
            
            # 数字を追加（白い背景付き）
            ax.text(x_pos, y_pos, f"{i+1}", ha='center', va='center', 
                   fontsize=6, color='black', fontweight='bold',
                   bbox=dict(facecolor='white', alpha=0.7, pad=0.1, boxstyle='round'))
        
        # カラーバーの追加
        cbar = plt.colorbar(im, ax=ax, fraction=0.02, pad=0.02)
        cbar.set_label('Feature Activation', fontsize=10)
        
        # グリッドの外枠を追加
        ax.set_frame_on(True)
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color('black')
            spine.set_linewidth(1.5)
        
        return ax
    
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

    def visualize_attention_overlay(self, image_path, attention_maps, layer_idx=-1, head_idx=0, alpha=0.5):
        """
        注意マップを元画像に重ね合わせて可視化する
        
        Args:
            image_path: 元画像のパス
            attention_maps: モデルから抽出された注意マップ
            layer_idx: 可視化するレイヤーのインデックス
            head_idx: 可視化するアテンションヘッドのインデックス
            alpha: オーバーレイの透明度（0-1）
        """
        # 画像の読み込み
        img = Image.open(image_path).convert('RGB')
        original_size = img.size
        img_np = np.array(img)
        
        # 注意マップの取得
        attn = attention_maps[layer_idx][0, head_idx].cpu().numpy()
        cls_attn = attn[0, 1:]  # CLS token -> patches
        
        # 画像グリッドに変換
        grid_size = int(np.sqrt(len(cls_attn)))
        cls_attn = cls_attn.reshape(grid_size, grid_size)
        
        # 注意マップの正規化（より明確なコントラストのために）
        cls_attn = (cls_attn - cls_attn.min()) / (cls_attn.max() - cls_attn.min() + 1e-8)
        
        # 注意マップを元画像サイズにリサイズ
        attention_resized = cv2.resize(cls_attn, original_size)
        
        # 注意マップをカラーマップに変換（より鮮明な色を使用）
        attention_colored = cv2.applyColorMap(
            (attention_resized * 255).astype(np.uint8),
            cv2.COLORMAP_JET  # VIRIDISからJETに変更
        )
        
        # BGRからRGBに変換
        attention_colored = cv2.cvtColor(attention_colored, cv2.COLOR_BGR2RGB)
        
        # オーバーレイ画像の作成（より強いコントラストのために）
        overlay = cv2.addWeighted(img_np, 1-alpha, attention_colored, alpha, 0)
        
        # 可視化
        fig, axes = plt.subplots(1, 3, figsize=(24, 8))
        
        axes[0].imshow(img_np)
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        axes[1].imshow(attention_colored)
        axes[1].set_title(f'Attention Map (Layer {layer_idx+1}, Head {head_idx+1})')
        axes[1].axis('off')
        
        axes[2].imshow(overlay)
        axes[2].set_title('Overlay')
        axes[2].axis('off')
        
        # カラーバーの追加
        norm = plt.Normalize(0, 1)
        sm = plt.cm.ScalarMappable(cmap=plt.cm.jet, norm=norm)
        sm.set_array([])
        plt.colorbar(sm, ax=axes[1], fraction=0.046, pad=0.04)
        
        plt.tight_layout()
        return fig

    def visualize_layer_features(self, image_path, layer_idx=0, num_features=64, figsize=(10, 10)):
        """
        特定のレイヤーの特徴を可視化する
        
        Args:
            image_path: 画像ファイルのパス
            layer_idx: 可視化するレイヤーのインデックス
            num_features: 可視化する特徴の数
            figsize: 出力画像のサイズ
        """
        # 特徴の抽出（中間層の特徴も取得）
        features, hidden_states = self.extract_features(image_path, return_hidden_states=True)
        
        # 指定されたレイヤーの特徴を取得
        layer_features = hidden_states[layer_idx]
        
        # 特徴の可視化
        fig = self.visualize_features(layer_features, num_features=num_features, figsize=figsize)
        plt.suptitle(f'Layer {layer_idx} Features', fontsize=16)
        return fig

    def visualize_multiple_layers(self, image_path, layer_indices=[0, 6, 11], num_features=64, figsize=(22, 24)):
        """
        複数のレイヤーの特徴を並べて可視化する
        
        Args:
            image_path: 画像ファイルのパス
            layer_indices: 可視化するレイヤーのインデックスのリスト
            num_features: 可視化する特徴の数
            figsize: 出力画像のサイズ
        """
        # 特徴の抽出（中間層の特徴も取得）
        features, hidden_states = self.extract_features(image_path, return_hidden_states=True)
        
        # 元画像の読み込み（参照用）
        orig_img = Image.open(image_path).convert('RGB')
        img_resized = orig_img.resize((224, 224))
        
        # 全体の図を作成（サブプロットを使用）
        fig = plt.figure(figsize=figsize)
        
        # グリッドスペックの設定（より柔軟なレイアウト）
        gs = fig.add_gridspec(len(layer_indices) + 2, 2, 
                             height_ratios=[1, 0.5] + [3] * len(layer_indices),
                             width_ratios=[1, 4])
        
        # 元画像を表示
        ax_img = fig.add_subplot(gs[0, :])
        ax_img.imshow(img_resized)
        ax_img.set_title('Original Image (224x224)', fontsize=18)
        ax_img.axis('off')
        
        # 説明テキストを追加
        ax_desc = fig.add_subplot(gs[1, :])
        ax_desc.axis('off')
        desc_text = (
            "Vision Transformer (ViT) Layer Features:\n"
            "• Low-level features (Layer 0): Detect basic visual elements like edges, colors, and textures\n"
            "• Mid-level features (Layer 6): Capture shapes, patterns, and parts of objects\n"
            "• High-level features (Layer 11): Represent semantic concepts and object identities"
        )
        ax_desc.text(0.5, 0.5, desc_text, ha='center', va='center', fontsize=14,
                   bbox=dict(facecolor='lightyellow', alpha=0.5, boxstyle='round,pad=1'))
        
        # 各レイヤーの特徴を表示
        layer_names = {
            0: "Low-level",
            6: "Mid-level",
            11: "High-level"
        }
        
        for idx, layer_idx in enumerate(layer_indices):
            # 指定されたレイヤーの特徴を取得
            layer_features = hidden_states[layer_idx]
            
            # レイヤーのタイトルとサムネイル
            ax_thumb = fig.add_subplot(gs[idx + 2, 0])
            
            # サムネイル画像を作成（特徴の平均）
            thumb_features = layer_features[0, 1:].mean(dim=1).cpu().numpy()
            size = int(np.sqrt(thumb_features.shape[0]))
            thumb_map = thumb_features.reshape(size, size)
            thumb_map = (thumb_map - thumb_map.min()) / (thumb_map.max() - thumb_map.min() + 1e-8)
            
            # サムネイル表示
            ax_thumb.imshow(thumb_map, cmap=f'viridis')
            
            # レイヤータイプに基づいて色付きのタイトルを設定
            if layer_idx == 0:
                color = 'green'
                desc = "Edges & Textures"
            elif layer_idx == len(hidden_states) - 1:
                color = 'red'
                desc = "Semantic Features"
            else:
                color = 'orange'
                desc = "Shapes & Patterns"
                
            ax_thumb.set_title(f"{layer_names.get(layer_idx, 'Layer')} {layer_idx}", 
                              fontsize=16, color=color, fontweight='bold')
            ax_thumb.set_xlabel(desc, fontsize=12)
            ax_thumb.axis('off')
            
            # 特徴の可視化（8x8グリッド）
            ax_grid = fig.add_subplot(gs[idx + 2, 1])
            ax_grid.axis('off')
            
            # 特徴の可視化（色分けあり）
            self._visualize_features_grid(layer_features, ax_grid, num_features=num_features, 
                                       cmap=f'viridis')
        
        # 全体のタイトル
        plt.suptitle('Vision Transformer Layer Features Comparison', fontsize=24, y=0.98)
        
        # レイアウト調整
        plt.subplots_adjust(hspace=0.4, top=0.95, bottom=0.05, left=0.05, right=0.95)
        
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
    feature_fig.savefig(f"{output_dir}/{img_name}_vit_features.png", dpi=300, bbox_inches='tight')
    plt.close(feature_fig)
    
    # 注意マップの可視化と保存（最後のレイヤーの最初のヘッド）
    attention_fig = extractor.visualize_attention(image_path, attention_maps)
    attention_fig.savefig(f"{output_dir}/{img_name}_vit_attention.png", dpi=300, bbox_inches='tight')
    plt.close(attention_fig)
    
    # オーバーレイ可視化の保存
    overlay_fig = extractor.visualize_attention_overlay(image_path, attention_maps)
    overlay_fig.savefig(f"{output_dir}/{img_name}_vit_attention_overlay.png", dpi=300, bbox_inches='tight')
    plt.close(overlay_fig)
    
    # 複数のレイヤーの特徴を可視化
    layer_fig = extractor.visualize_multiple_layers(image_path, layer_indices=[0, 6, 11])
    layer_fig.savefig(f"{output_dir}/{img_name}_vit_layer_features.png", dpi=300, bbox_inches='tight')
    
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