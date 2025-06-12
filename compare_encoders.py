"""
画像エンコーダー比較スクリプト

このスクリプトでは、異なる画像エンコーダー（ViT、CLIP、DETR）を使って
同一の高解像度画像（1024×1024）を処理し、各モデルの特徴抽出能力を比較します。
"""

import os
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import time

# 各モデルのインポート
from models.vit import process_high_resolution_image
from models.clip import process_high_resolution_image_with_clip
from models.detr import process_high_resolution_image_with_detr


def create_comparison_report(image_path, output_dir='outputs'):
    """
    各画像エンコーダーを使って画像を処理し、結果を比較するレポートを作成する

    Args:
        image_path: 処理する画像のパス
        output_dir: 結果を保存するディレクトリ
    """
    # 出力ディレクトリの作成
    os.makedirs(output_dir, exist_ok=True)
    
    # 画像ファイル名の取得
    img_name = Path(image_path).stem
    
    # 元画像の表示
    img = Image.open(image_path).convert('RGB')
    img_w, img_h = img.size
    print(f"処理する画像: {image_path}, サイズ: {img_w}x{img_h}")
    
    # 各モデルの実行時間を測定
    models = ['ViT', 'CLIP', 'DETR']
    execution_times = []
    
    # Vision Transformer
    print("\n1. Vision Transformer (ViT)による特徴抽出を開始...")
    start_time = time.time()
    vit_features = process_high_resolution_image(image_path, output_dir)
    vit_time = time.time() - start_time
    execution_times.append(vit_time)
    print(f"ViT処理時間: {vit_time:.2f}秒")
    
    # CLIP
    print("\n2. CLIPによる特徴抽出を開始...")
    start_time = time.time()
    clip_features = process_high_resolution_image_with_clip(image_path, output_dir)
    clip_time = time.time() - start_time
    execution_times.append(clip_time)
    print(f"CLIP処理時間: {clip_time:.2f}秒")
    
    # DETR
    print("\n3. DETRによる物体検出と特徴抽出を開始...")
    start_time = time.time()
    detr_results = process_high_resolution_image_with_detr(image_path, output_dir)
    detr_features, boxes, labels, scores = detr_results
    detr_time = time.time() - start_time
    execution_times.append(detr_time)
    print(f"DETR処理時間: {detr_time:.2f}秒")
    
    # 実行時間の比較グラフ
    plt.figure(figsize=(10, 6))
    plt.bar(models, execution_times)
    plt.xlabel('モデル')
    plt.ylabel('実行時間 (秒)')
    plt.title('各画像エンコーダーの実行時間比較')
    plt.savefig(f"{output_dir}/{img_name}_execution_time_comparison.png")
    
    # レポートの作成
    with open(f"{output_dir}/{img_name}_comparison_report.md", 'w', encoding='utf-8') as f:
        f.write(f"# 画像エンコーダー比較レポート: {img_name}\n\n")
        
        f.write("## 処理画像\n")
        f.write(f"- ファイル名: {Path(image_path).name}\n")
        f.write(f"- サイズ: {img_w}x{img_h}\n\n")
        
        f.write("## 各モデルの実行時間\n")
        for model, exec_time in zip(models, execution_times):
            f.write(f"- {model}: {exec_time:.2f}秒\n")
        f.write("\n")
        
        f.write("## 各モデルの特徴\n\n")
        
        f.write("### 1. Vision Transformer (ViT)\n")
        f.write("- **特徴量の次元**: " + str(vit_features.shape) + "\n")
        f.write("- **特徴抽出の特性**: グローバルな文脈情報を捉えた特徴表現\n")
        f.write("- **可視化結果**: [特徴マップ](./" + f"{img_name}_vit_features.png), ")
        f.write("[注意マップ](./" + f"{img_name}_vit_attention.png), ")
        f.write("[複数注意ヘッド](./" + f"{img_name}_vit_multi_attention.png)\n\n")
        
        f.write("### 2. CLIP\n")
        f.write("- **特徴量の次元**: " + str(clip_features.shape) + "\n")
        f.write("- **特徴抽出の特性**: 言語と画像の共有埋め込み空間での特徴表現\n")
        f.write("- **可視化結果**: [特徴分布](./" + f"{img_name}_clip_features.png), ")
        f.write("[テキスト類似度](./" + f"{img_name}_clip_similarities.png), ")
        f.write("[カテゴリ類似度](./" + f"{img_name}_clip_categories.png)\n\n")
        
        f.write("### 3. DETR\n")
        f.write("- **特徴量の次元**: " + str(detr_features.shape) + "\n")
        f.write("- **検出された物体数**: " + str(len(labels)) + "\n")
        
        if len(labels) > 0:
            # クラスごとの物体数をカウント
            unique_labels, counts = np.unique(labels, return_counts=True)
            class_names = [f"{COCO_CLASSES[label]} ({count}個)" for label, count in zip(unique_labels, counts)]
            f.write("- **検出された物体クラス**: " + ", ".join(class_names) + "\n")
        
        f.write("- **可視化結果**: [物体検出](./" + f"{img_name}_detr_detection.png), ")
        f.write("[特徴マップ](./" + f"{img_name}_detr_features.png)")
        
        if len(labels) > 0:
            f.write(", [物体分布](./" + f"{img_name}_detr_class_distribution.png)")
        
        f.write("\n\n")
        
        f.write("## 考察と比較\n\n")
        f.write("### 特徴抽出能力の比較\n")
        f.write("- **ViT**: 画像全体のグローバルな特徴を捉え、自己注意機構により画像内の関連性を抽出します。\n")
        f.write("- **CLIP**: 言語的な意味を持つ特徴を抽出し、テキスト記述との類似性を通じて意味論的な理解を示します。\n")
        f.write("- **DETR**: 物体中心の特徴表現を行い、物体の位置や種類に関する情報を詳細に抽出します。\n\n")
        
        f.write("### 計算効率\n")
        fastest_model = models[np.argmin(execution_times)]
        f.write(f"- 最も高速なモデル: **{fastest_model}** ({min(execution_times):.2f}秒)\n")
        f.write(f"- 最も遅いモデル: **{models[np.argmax(execution_times)]}** ({max(execution_times):.2f}秒)\n\n")
        
        f.write("### 研究応用への推奨\n")
        f.write("- **画像分類タスク**: ViTまたはCLIP\n")
        f.write("- **物体検出タスク**: DETR\n")
        f.write("- **テキスト関連タスク**: CLIP\n")
        f.write("- **高解像度画像解析**: タスクに応じて選択（物体検出ならDETR、全体的な特徴ならViT）\n")
    
    print(f"\n比較レポートが作成されました: {output_dir}/{img_name}_comparison_report.md")


if __name__ == "__main__":
    # COCO クラスのインポート（DETRレポート用）
    from models.detr import COCO_CLASSES
    
    parser = argparse.ArgumentParser(description='画像エンコーダー比較')
    parser.add_argument('--image', type=str, default='data/00325_hr.png',
                        help='処理する画像のパス')
    parser.add_argument('--output', type=str, default='outputs',
                        help='結果を保存するディレクトリ')
    
    args = parser.parse_args()
    
    # 比較レポートの作成
    create_comparison_report(args.image, args.output) 