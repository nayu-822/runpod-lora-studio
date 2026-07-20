# RunPod LoRA Studio

RunPod GPU Pod 上で動かす、SDXL LoRA 向けの Gradio ベース作業アプリです。  
このリポジトリの現状は Phase 1 の基盤実装で、プロジェクト管理とローカル画像登録までを含みます。

## Phase 1 の範囲

- `src` レイアウトの Python パッケージ化
- `python -m runpod_lora_studio.app` で起動できる最小 Gradio UI
- `.env` / 環境変数ベースの設定ロード
- RunPod 向けの環境検証スクリプト
- 起動スクリプトと bootstrap スクリプト
- `ruff` / `mypy` / `pytest` / GitHub Actions の整備
- SQLite、SQLAlchemy、Alembicによるプロジェクト・画像メタデータ保存
- Pillowによる画像実体検証、SHA-256計算、サムネイル生成
- 採用・保留・除外の状態管理

## 要件

- Python 3.11 以上
- Linux / RunPod 環境を想定
- GPU は本番向け前提

ローカルの CPU 環境でも、アプリ import、設定ロード、テスト、環境検証スクリプトの実行はできる構成です。

## セットアップ

`.env.example` をもとに `.env` を作成します。

```bash
cp .env.example .env
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
alembic upgrade head
```

`alembic upgrade head`は設定されたデータベースへ初期スキーマを適用します。アプリ起動時にテーブルを無条件作成することはありません。

## ローカル起動

```bash
python -m runpod_lora_studio.app
```

Gradio は既定で `0.0.0.0:7860` で起動します。ローカル確認では `http://127.0.0.1:7860` を開いてください。

## RunPod での起動

```bash
bash scripts/bootstrap_runpod.sh
bash scripts/start.sh
```

RunPod 側では HTTP Proxy 経由で 7860 番ポートに接続する想定です。

## 環境検証

```bash
python scripts/verify_environment.py
```

人間向けの要約を標準出力へ表示し、機械処理が必要な場合だけ次のようにJSON形式を指定します。

```bash
python scripts/verify_environment.py --json
```

このスクリプトは次を表示します。

- Python / OS 情報
- `RUNPOD_POD_ID` の有無
- PyTorch / CUDA / bf16 対応情報
- 複数GPUのインデックス、名前、VRAM
- 設定された作業ディレクトリの書き込み可能性とディスク容量
- `git` / `rclone` / `nvidia-smi` の有無（`git`は必須、その他は警告）

ローカル CPU 環境では、未検出項目を警告として表示します。
必須条件に失敗した場合だけ終了コード1になります。APIキーなどの秘密情報は表示しません。

## テストと静的チェック

```bash
ruff format --check .
ruff check .
mypy src
pytest
```

## 現時点で未実装の内容

Phase 1 では、以下はまだ実装していません。

- pHashによる近似重複判定、品質評価
- Danbooru 連携
- タグ付け
- 学習ジョブ管理
- Google Drive 同期
- RunPod Pod の Stop / Terminate 制御

## セキュリティメモ

- `.env` や API キーはコミットしないでください
- `RUNPOD_API_KEY` は将来の機能用で、現段階では利用しません
- `rclone.conf` や学習成果物は Git 管理対象外です

## ローカル Phase 1 確認

Windows ネイティブ、WSL2、Linux のいずれでも、Python 3.11 以上の仮想環境で確認できます。

```powershell
Copy-Item .env.local.example .env.local
& .\.venv\Scripts\python.exe -m pip install -e ".[dev]"
$env:RUNPOD_LORA_STUDIO_ENV_FILE = ".env.local"
& .\.venv\Scripts\alembic.exe upgrade head
& .\.venv\Scripts\python.exe scripts\verify_local_environment.py
& .\.venv\Scripts\python.exe -m runpod_lora_studio.app
```

Linux または WSL2 では、仮想環境を有効化した後に `cp .env.local.example .env.local`、
`RUNPOD_LORA_STUDIO_ENV_FILE=.env.local alembic upgrade head`、
`RUNPOD_LORA_STUDIO_ENV_FILE=.env.local python scripts/verify_local_environment.py` を実行します。
UI は `127.0.0.1:7860` から確認します。JPEG、PNG、WebP を登録でき、再起動後も SQLite のプロジェクトと画像状態が復元されます。

PyTorch は Phase 1 の必須依存ではありません。未導入または CPU 版でもローカル確認は成立し、`torch.cuda.is_available()` が `False` であることは正常です。
GPU と CUDA の確認は RunPod 上で行います。rclone も任意で、必要な場合だけ `rclone version`、`rclone listremotes`、`rclone lsd <remote>:<path>` で設定を確認します。
Phase 1 では同期処理を実行せず、認証情報や `rclone.conf` を Git に保存しません。
