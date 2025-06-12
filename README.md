# image_encoder
画像エンコーダーについて勉強するためのリポジトリ

## 実装した画像エンコーダー

### 1. Vision Transformer (ViT)
- 画像をパッチに分割し、Transformerで処理するモデル
- CNNとは異なりグローバルな文脈を捉えることが可能
- 実装: `models/vit.py`

### 2. CLIP (Contrastive Language-Image Pre-training)
- OpenAIが開発した画像とテキストの対応関係を学習するモデル
- Zero-shot分類などに優れた性能を示す
- 実装: `models/clip.py`

### 3. 物体検出モデル
- DETR (DEtection TRansformer)
- Transformerベースの新しい物体検出アプローチ
- 実装: `models/detr.py`

## セットアップ

### 必要環境
- Python 3.8以上
- PyTorch 1.9.0以上
- CUDA対応GPU（推奨）

### インストール手順

1. リポジトリのクローン
```bash
git clone https://github.com/yokochi312/image_encoder.git
cd image_encoder
```

2. 依存パッケージのインストール
```bash
pip install -r requirements.txt
```

3. CLIPのインストール（PyTorchHubから自動的にダウンロードされない場合）
```bash
pip install git+https://github.com/openai/CLIP.git
```

## 使用方法

### 1. 個別のエンコーダーを実行

#### Vision Transformer
```bash
python models/vit.py --image data/00325_hr.png --output outputs
```

#### CLIP
```bash
python models/clip.py --image data/00325_hr.png --output outputs
```

#### DETR
```bash
python models/detr.py --image data/00325_hr.png --output outputs --threshold 0.7
```

### 2. すべてのエンコーダーを比較

すべてのエンコーダーを一度に実行し、比較レポートを生成します。

```bash
python compare_encoders.py --image data/00325_hr.png --output outputs
```

出力ディレクトリには以下のファイルが生成されます：
- 各エンコーダーの特徴マップ可視化結果
- 実行時間比較グラフ
- 詳細な比較レポート（Markdown形式）

## 特徴抽出の詳細

各エンコーダーが抽出できる特徴の詳細については、以下のドキュメントを参照してください：
- [画像エンコーダーの特徴抽出能力と研究応用](./image_encoder_features.md)

## ディレクトリ構造

```
image_encoder/
├── models/
│   ├── vit.py        # Vision Transformer実装
│   ├── clip.py       # CLIP実装
│   └── detr.py       # DETR実装
├── data/             # サンプル画像
├── outputs/          # 出力結果（自動生成）
├── compare_encoders.py  # エンコーダー比較スクリプト
├── image_encoder_features.md  # 特徴抽出能力の詳細解説
├── requirements.txt  # 依存パッケージリスト
└── README.md         # このファイル
```

## 対応画像サイズ

基本的には1024×1024サイズの高解像度画像を処理対象としていますが、各エンコーダーは内部で適切にリサイズして処理します。どのような解像度の画像でも入力可能です。

## 参考文献・リソース

- ViT: [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929)
- CLIP: [Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020)
- DETR: [End-to-End Object Detection with Transformers](https://arxiv.org/abs/2005.12872)
