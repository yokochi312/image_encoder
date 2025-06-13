# image_encoder
画像エンコーダーについて勉強するためのリポジトリ

## 実装した画像エンコーダー

### 1. Vision Transformer (ViT)
- 画像をパッチに分割し、Transformerで処理するモデル
- CNNとは異なりグローバルな文脈を捉えることが可能
- 実装: `models/vit.py`

#### ViTの特徴抽出と可視化
ViTモデルは以下のような特徴を抽出し、可視化することができます：

1. **特徴の構造**:
- サイズ: `[batch_size, sequence_length, hidden_size]`
- デフォルトのViT-baseモデルでは：
  - `hidden_size = 768`（特徴ベクトルの次元）
  - `sequence_length = 197`（CLSトークン + 196パッチ）

2. **特徴の種類**:
- **CLSトークンの特徴**:
  - 画像全体の情報を集約した特徴
  - 分類タスクで使用される主要な特徴

- **パッチの特徴**:
  - 各パッチ（16x16ピクセル）の特徴
  - 画像の局所的な情報を表現

3. **レイヤーごとの特徴**:
- **低レイヤー（浅い層）**:
  - エッジ、テクスチャ、色などの基本的な視覚的特徴
  - より局所的な情報
  - 例：毛の質感、耳の輪郭、目の縁取りなど

- **高レイヤー（深い層）**:
  - 物体の形状、パターン、意味的な情報
  - より抽象的な表現
  - 例：猫全体のシルエット、顔の配置、「猫らしさ」を表す特徴など

4. **可視化機能**:
- 特徴マップの可視化（64個の特徴を8×8グリッドで表示）
- 注意マップの可視化（モデルが注目している領域を表示）
- オーバーレイ可視化（注意マップを元画像に重ね合わせ）
- 複数レイヤーの特徴比較（低レイヤーから高レイヤーまでの変化を確認）

#### 出力ファイル
ViTモデルを実行すると、以下のファイルが生成されます：
- `{画像名}_vit_features.png`: 最終層の特徴マップ
- `{画像名}_vit_attention.png`: 注意マップ
- `{画像名}_vit_attention_overlay.png`: 注意マップのオーバーレイ
- `{画像名}_vit_layer_features.png`: 異なるレイヤーの特徴比較

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
