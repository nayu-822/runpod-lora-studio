# RunPod LoRA Studio

RunPod GPU Pod 上で動かす、SDXL LoRA 向けの Gradio ベース作業アプリです。  
このリポジトリの現状は Phase 0 の基盤実装で、アプリ本体の最小起動、設定管理、環境検証、テスト・CI の土台までを含みます。

## Phase 0 の範囲

- `src` レイアウトの Python パッケージ化
- `python -m runpod_lora_studio.app` で起動できる最小 Gradio UI
- `.env` / 環境変数ベースの設定ロード
- RunPod 向けの環境検証スクリプト
- 起動スクリプトと bootstrap スクリプト
- `ruff` / `mypy` / `pytest` / GitHub Actions の整備

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
```

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

Phase 0 では、以下はまだ実装していません。

- SQLite 永続化本体
- Danbooru 連携
- タグ付け
- 学習ジョブ管理
- Google Drive 同期
- RunPod Pod の Stop / Terminate 制御

## セキュリティメモ

- `.env` や API キーはコミットしないでください
- `RUNPOD_API_KEY` は将来の機能用で、現段階では利用しません
- `rclone.conf` や学習成果物は Git 管理対象外です
