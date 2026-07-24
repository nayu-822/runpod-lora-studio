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

### Phase 1の最終確認事項

- ローカル環境確認ではGitを必須項目として確認します。Git未導入や実行失敗でも、SQLite、Alembic、任意項目の確認を最後まで継続し、最終終了コードは非0になります。
- ポート確認は固定値ではなく、設定された`gradio_server_port`を対象にします。
- サムネイルの一時ファイル清掃に失敗しても、元の「サムネイル生成に失敗しました。」という安全なメッセージを維持します。
- Phase 1の通常テスト・CIではGoogle Driveへ実通信しません。

### Phase 1のUI・障害時確認

- プロジェクト一覧を再読込すると、表示中のドロップダウンと内部選択を同期します。現在のプロジェクトが残っていれば維持し、存在しない場合は先頭へ移り、0件なら未選択になります。
- 画像0件またはフィルター結果0件は `0 / 0ページ、全0件` と表示します。
- 画像検査ではピクセル数上限とPillowのDecompression Bomb相当の異常を安全な日本語メッセージで拒否します。
- DB登録失敗やサムネイル生成失敗時は、原画像、サムネイル、一時ファイルを清掃し、途中のDB行を残しません。
- ローカル環境確認は実運用DBを変更せず、書き込み確認を一時SQLite DBで行います。通常のテストやCIからGoogle Driveへアクセスしません。

### Phase 1の確認上の注意

- 画像登録結果には成功件数、失敗件数、重複の参考警告件数、ファイルごとの安全な失敗理由が表示されます。
- 同一SHA-256の画像は自動削除・自動除外せず、参考警告だけを表示します。
- Galleryのサムネイル選択は単一画像の対象になり、複数画像の一括変更は下のチェックボックスを使用します。ページ・検索・状態変更時に古い選択をクリアします。
- `verify_local_environment.py` は実運用DBへ`SELECT 1`だけを行い、書き込み確認はランダム名の一時SQLite DBで行って確認後に削除します。
- ローカル確認ではAlembicの`current`と`head`が一致していることを必須条件として判定します。未適用DBや古いリビジョンは失敗になります。
- PyTorchとrcloneは任意です。rclone未導入・未設定は任意状態、導入済みの実行失敗は警告として扱います。通常のテストやCIからGoogle Driveへ接続しません。

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
## Phase 2A: Image inspection

画像タブの「プロジェクト全体を検査」で、登録済み画像を1枚ずつ検査できます。
「選択画像を再検査」はチェックした画像だけを対象にします。結果はDBへ現在値として保存され、
同じ画像・検査器バージョンの結果が無制限に増えることはありません。

検査項目はSHA-256完全重複、最低解像度、極端な縦横比、低情報量候補、ぼけ候補です。
完全重複の代表画像は登録日時が最も古いもの、同時刻なら画像IDの昇順で決定します。
原画像は削除・上書きせず、検査の再実行で採用・保留・除外の手動状態を変更しません。

デフォルト閾値は最低幅・高さ512px、最大縦横比3.0、低情報量のグレースケール標準偏差8.0、
ぼけの鮮明度スコア50.0です。ぼけスコアはOpenCVを使わず、縮小グレースケール画像の
3x3 Laplacian varianceを計算し、大きいほど鮮明と解釈します。低情報量とぼけは誤判定の可能性があるため、
自動除外ではなく確認用のwarningです。閾値は`RUNPOD_LORA_STUDIO_INSPECTION_*`で変更できます。

画像ファイルの消失・破損は該当画像の検査失敗として記録し、他の画像の検査は継続します。
近似重複（pHash）と類似画像比較UIはPhase 2Bの対象であり、Phase 2Aには含まれません。
