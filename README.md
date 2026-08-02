# RunPod LoRA Studio

RunPod GPU Pod 上で動かす、SDXL LoRA 向けの Gradio ベース作業アプリです。  
このリポジトリは、Phase 1〜4の画像選別・タグ付け・固定スナップショットに加え、Phase 5のモデル管理とGoogle Drive連携基盤を含みます。

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

## GPU identity と学習時メモリ測定

TorchのGPUDeviceInfo.indexは、CUDA_VISIBLE_DEVICES適用後のlogical indexです。
physical indexとは直接比較せず、固定queryのnvidia-smi inventory（index、UUID、GPU名、
total VRAM、compute capability）とUUIDを照合します。CUDA_VISIBLE_DEVICES=2,0は
logical 0 → physical 2、logical 1 → physical 0と解決し、UUID指定は大文字小文字を
正規化した一意のexact/prefix matchだけを受け付けます。不正・曖昧・不一致なtokenは
架空のfingerprintを作らずwarningまたはfailed snapshotにします。

torch.cuda.mem_get_info()は(free, total)の順で読み取り、total VRAMはfree VRAMの
変動から独立して保存します。開始前にGPUを一意に確定できない複数GPUジョブでは、
対象PIDのcompute-process queryで単一GPU UUIDを確認できた場合だけruntime identityを
別スナップショットへ保存し、summaryとcalibrationはそのidentity由来の物理GPU属性を
使用します。推奨付きジョブで実行GPUを確定できない場合は開始を拒否します。
最初に確定したselected GPU identityは学習中も上書きせず、異なるUUIDを観測した場合は
変更監査情報を保存し、そのジョブを速度・VRAM校正から除外します。

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
検査結果は検査器バージョンごとに保持し、再検査では対象バージョンだけを置き換えます。
画面とサマリーは現在の検査器バージョンのみを表示します。
近似重複（pHash）と類似画像比較UIはPhase 2Bの対象であり、Phase 2Aには含まれません。

Phase 2Aの検査対象はPhase 2Bでkeyset paginationのバッチ処理へ移行しました。比較時に保持するのはpHashと代表選定に必要な最小メタデータで、画像ファイルは1枚ずつ閉じます。
## Phase 2B: pHash近似重複検出

pHash（perceptual hash）は、画像の見た目を縮約した固定長ハッシュです。SHA-256の完全一致とは異なり、PNG/JPEG再圧縮や軽微な変更をハミング距離で比較できます。今回の実装はPillowでEXIF Orientationを補正し、透過画像を白背景へ合成してRGBへ正規化した後、`imagehash.phash`で計算します。

- 既定値: `hash_size=8`（64 bit）、最大ハミング距離 `8`。距離が閾値と等しいペアも候補に含めます。
- 保存形式: 小文字の固定長16進文字列（hash_size=8なら16文字）。algorithm、hash_size、detector_version、状態、日時も同じレコードへ保存します。
- グループ形成: 同じalgorithm/hash_size/detector_versionのペアだけを比較し、閾値以下の無向辺の連結成分をグループにします。連鎖による過剰なグループ化を確認できるよう、代表画像との距離とグループ内最小距離を表示します。代表画像自身の代表距離は常に`0`です。
- 代表候補: failedでない画像、警告数が少ない画像、高解像度、ぼけスコア、完全重複の既存代表を順に考慮します。同点はcreated_atとimage_idで決定的に解決し、手動代表は再検査でも維持します。複数の旧グループの手動代表が1グループへ統合された場合は、旧グループのcreated_atが古い候補、次に画像のcreated_at、最後にimage_idの昇順で1件を選びます。候補が現在利用できない場合は決定的な自動代表へ戻します。
- 「完全重複」は既存のSHA-256検査、「近似重複」はpHash検査としてUIとDBで区別します。類似候補は候補提示に過ぎず、自動削除や自動除外は行いません。手動で「類似ではない」とした関係は正規化したimage_idペアとして保存し、再グループ化時の直接辺から除外します。別の経路でつながる場合は否定済みペアが同じ連結成分に残る可能性があり、画面には否定済みペアを表示します。グループ全体を否定すると全ペアが否定され、次回再検査で直接辺が除外されてグループが解体されます。
- `RUNPOD_LORA_STUDIO_PHASH_HASH_SIZE`、`RUNPOD_LORA_STUDIO_PHASH_DISTANCE_THRESHOLD`、`RUNPOD_LORA_STUDIO_PHASH_BATCH_SIZE`、`RUNPOD_LORA_STUDIO_SIMILARITY_GROUP_PAGE_SIZE`で設定できます。画像ファイルは1枚ずつ開き、プロジェクト全体はkeyset paginationのバッチで処理します。
- 現在の比較はpHash値と代表選定に必要なメタデータだけをメモリへ保持するO(n²)比較です。想定上限は数千枚程度、1000枚では約50万ペアです。数万枚を扱う場合はBK-tree/LSH等の近傍探索を導入することが将来条件です。
- 類似グループ画面では、グループ一覧、サムネイル横並び比較、代表変更、類似確認／否定、採用・保留・除外を操作できます。原画像とPhase 2A検査結果は変更しません。
- CLIP埋め込みによる意味的類似判定、クロップ違い・顔・OCR等は未実装です。

## Phase 3: 自動タグ付け・手動タグ選別・最終キャプション編集

Phase 3では、採用（`accepted`）画像だけを対象にWD Tagger互換のアダプターでタグを推論し、TaggerRun、画像単位の結果、構造化タグ、最終キャプション、編集履歴をSQLiteへ保存します。通常のテストやローカル起動では外部モデルをダウンロードしません。既定のモデル識別子は`SmilingWolf/wd-eva02-large-tagger-v3`、revisionは`main`、保存先は`TAGGER_MODEL_DIR`です。RunPodではモデルファイルとonnxruntime・NumPyをこのディレクトリへ用意します。`TAGGER_ALLOW_MODEL_DOWNLOAD=true`を明示した場合だけ、huggingface_hubで指定revisionを一時ディレクトリへ取得し、`model.onnx`と`selected_tags.csv`を検証してから確定ディレクトリへ移動します。

前処理はEXIF Orientation補正、RGBAの白背景合成、RGB化、正方形contain、448px入力です。原画像は上書きしません。`auto`はCUDAが利用できればCUDA、できなければCPUへフォールバックします。既定閾値はgeneral 0.35、character 0.85で、閾値を上げるほど低信頼タグが減り、下げるほど取りこぼしが減る代わりにノイズが増えます。rating、character、generalをカテゴリとして保持し、raw・normalized・display表現を分離します。

実行モードは未処理のみ、失敗のみ、採用画像全件です。既存の検出結果や手動キャプションを自動的に上書きせず、失敗画像があっても他画像を継続します。キャンセル、異常終了、stale復旧をTaggerRun状態として記録します。タグ頻度は画像ごとに重複を1回へ畳み、対象画像数を分母として出現数降順、出現率降順、カテゴリ名、正規化名で決定的に並べます。

保持チェックは草稿として扱い、適用前に変更前後、除外タグ、対象画像数、警告、設定スナップショットを含むプレビューを生成します。プレビューは署名トークンで検証し、ルール保存とキャプション更新を同一トランザクションで行います。手動キャプションの扱いは保持、生成元から再構築、手動画像を除外の3ポリシーです。トリガーワードはタグと分離し、正規化・重複除去後に先頭へ置きます。各編集は履歴へ保存し、生成元または直前状態へ復元できます。

設定例は`.env.example`と`.env.local.example`にあります。Phase 3ではデータセットTOMLの生成、学習実行、成果物同期は行わず、これらはPhase 4以降の対象です。

TaggerRunの`target_image_count`はRun開始時点で実際に推論する画像数です。既存結果などで対象外になった画像は`skipped_image_count`へ分けて保存し、頻度集計の`used_image_count`は現在acceptedで、該当Runのcompleted結果かつタグが1件以上ある画像数です。partially_failed Runも利用できますが、失敗画像は頻度集計・キャプション生成の対象外です。

Run開始時には採用画像のIDとパスをメモリへ保持して対象集合を固定します。画像本体は一括ロードせず、処理時に1枚ずつ開いて閉じます。現在は数千枚程度を想定しており、将来はRun対象画像テーブルまたはkeyset paginationによるワーカー側ストリーミングへ移行します。Run開始後のSelectionState変更は実行中の対象集合には反映せず、キャプションプレビューでは現在のaccepted集合・状態・キャプションrevisionが変化していれば適用を拒否します。

## Phase 4: 学習用データセットスナップショット

Phase 4では、学習開始前の採用画像とその時点のcurrentキャプションを、後からプロジェクトを編集しても変わらない独立成果物として固定します。スナップショット作成は学習やGoogle Drive同期を開始せず、RunPod内のプロジェクト領域へ保存します。

保存先は次の構成です。画像とキャプションのファイル名は元ファイル名を使わず、決定的な連番と安全な拡張子で生成します。

```text
projects/{project_id}/dataset_snapshots/{snapshot_id}/
├── images/000001.png
├── captions/000001.txt
├── configs/dataset.toml
├── reports/dataset_report.json
├── reports/dataset_report.md
├── reports/tag_frequency.csv
├── reports/resolution_distribution.csv
├── reports/aspect_ratio_distribution.csv
├── reports/warnings.json
├── manifest.json
└── snapshot.json
```

プレビューではaccepted画像、SelectionState、原画像のSHA-256とファイルサイズ、currentキャプションのID・revision・本文ハッシュ、元TaggerRun、トリガーワード、DatasetSettings、生成器バージョンを署名します。適用時にDBから同じ集合を再取得し、いずれかが変わっていれば「プレビューの有効期限が切れています。再生成してください。」として書き込みを開始しません。対象画像はacceptedのみです。

currentキャプションがない、原画像がない・読めない、DB保存ハッシュまたはサイズと一致しない、設定・TOML・コピー後ハッシュ・manifest・最終検証に失敗した場合は必須エラーとして作成を停止します。品質検査、完全重複、pHash近似重複、トリガーワード不足、空ではないが構造化タグがない状態は警告として表示し、警告確認を明示した場合だけ作成できます。品質・重複警告だけで自動的に画像を除外しません。

キャプションファイルはUTF-8（BOMなし）、LF改行、末尾改行1つへ正規化します。TOMLはSDXL向けのresolution、bucket、caption、augmentation、repeats設定を生成し、`tomllib`で再パースしてから保存します。既定値はresolution 1024、min/max bucket 256/2048、steps 64、num_repeats 1、caption extension `.txt`です。`DatasetConfigService`は範囲、整列、拡張子、制御文字、空subset、極端なrepeatsを検査します。

`manifest.json`には画像・キャプションの対応、元／スナップショットのハッシュ、サイズ、解像度、タグ数、トリガーワード数、品質・重複状態、警告、設定、TOMLハッシュを記録します。`content_sha256`は、sequence、画像相対パス、キャプション相対パス、スナップショット画像SHA-256、キャプションSHA-256をsequence順に並べ、`dataset_toml_sha256`と正規化設定JSONのSHA-256を加えたpayloadから計算します。絶対パス、作成日時、snapshot IDは含めません。レポートには解像度（短辺・長辺・総画素数のmin/max/mean/median/p10/p25/p75/p90と境界別件数）、縦横比、bucket候補、画像単位重複率と近似重複率、画像単位タグ頻度、トリガーワード付与率、空キャプション数、警告を出力します。同一キャプション内のタグ重複は1回として集計します。

DBには`dataset_snapshots`、`dataset_snapshot_items`、`dataset_validation_issues`、`snapshot_creation_jobs`を追加し、Alembic 0006で作成します。状態は`draft`、`validating`、`creating`、`db_finalization_pending`、`completed`、`failed`、`canceled`、`corrupted`です。作成中は`{snapshot_id}.creating`へ書き込み、全ファイル・TOML・manifest・レポート・ハッシュ検証完了後に同一プロジェクト内へatomic renameし、DBをcompletedへ更新します。rename後のDB保存・commitに失敗した場合は確定ファイルを削除せず`db_finalization_pending`へ記録し、起動時または回復操作でmanifestからitem/issueを冪等に再構築します。コピーは固定バッファのストリーミングと画像単位のキャンセル確認を行い、再起動時にcreatingのstale行はfailedへ復旧します。再検証で破損を検出した場合はcorruptedへ変更しますが、スナップショットや原画像を削除しません。

現在のUIはプレビュー、警告確認付き作成、一覧、再検証を提供します。作成完了後に原画像・キャプション・SelectionStateを変更してもスナップショットは変化しません。スナップショットは将来のPhase 5/6で学習入力と同期対象として利用しますが、Phase 4単体では学習実行、成果物同期、削除機能は提供しません。

画像本体は一括読込せず、`.part`へ固定サイズバッファでコピーし、SHA-256とサイズ検証後に確定名へrenameします。容量検査では画像、キャプション、manifest/report/CSV/TOML等のメタデータ、設定可能な安全マージンを合算し、保存先filesystemの空き容量不足をDB作成前に拒否します。現在はプレビューのaccepted画像についてIDとパス、キャプションメタデータをメモリへ保持する方式で、想定上限は数千枚程度です。将来はスナップショット対象テーブルのkeyset paginationとストリーミング作成へ移行し、アプリ再起動後も対象集合をDBから再現できるようにします。

類似グループは対象画像に関係するグループをID単位で一度だけ集計します。review status、全体member count、代表画像、否定済みペア数を参照し、未確認グループ数、否定済みペアを含むグループ数、未確認グループの対象画像数をプレビューとレポートへ表示します。未確認・否定済みペアを理由に自動除外はしません。

### 転送検証ポリシー

`full_checksum`はremote SHA-256とローカルSHA-256を比較し、MD5だけでは成功にしません。`remote_hash_and_size`はremoteが提供する比較可能なhashとサイズを使います。`size_and_manifest`はファイルの存在・サイズ・manifestに保存したremoteサイズ・更新日時・hashメタデータ・snapshot content SHA-256・必須ファイル集合を確認する限定検証です。`existence_only`は存在だけを確認します。

`storage_remote_hash_fallback`は実処理へ反映されます。`error`は失敗、`existence_only`は`existence_only`へ、`size_and_manifest`は`manifest_metadata_and_size`へフォールバックします。設定値はmanifestのsettingsへ、実績は各itemの`verification_status`と全体`verification_level`へ保存します。全体レベルはitem実績から決定し、`not_verified`／`verification_failed`を含むジョブはcompletedにしません。

`size_and_manifest`では有効なtransfer-manifest.jsonと対象itemが必須で、remoteサイズ・更新日時・hashメタデータ・snapshot content SHA-256を照合します。`existence_only`はremote内容の同一性を意味せず、skip_identicalの判定には使用しません。転送後検証でのみ存在確認として成功でき、全体verification levelは最も弱いitemの水準になります。

`size_and_manifest`では、仮manifestでremoteのメタデータを検証した後、各itemの検証結果を反映した最終manifestをremoteへ再アップロードします。ジョブをcompletedにする前に、最終manifestをremoteから読み戻し、schema、ジョブ・プロジェクト・snapshot、content SHA-256、item集合、転送状態、検証状態、サイズ、ローカルSHA-256がローカルの最終版と一致することを確認します。最終manifestが欠落・破損・仮版のまま、または別ジョブの内容の場合は、転送済みファイルを削除せずジョブを失敗扱いにします。

0009で追加した累積進捗列は既存runningジョブでは0から初期化されます。既存ジョブのheartbeat／PIDは変更せず、再起動時のstale回復対象となった場合は従来どおりstaleへ遷移します。新しい転送は完了済みバイトと現在ファイルバイトを分けて更新します。

## Phase 5: モデル管理・Google Drive・rclone連携

Phase 5では、rcloneを介してGoogle Driveのモデルを一覧表示し、RunPodのローカルキャッシュへ検証付きで取得できます。モデル一覧は許可拡張子、サイズ上限、ページング、検索条件で絞り込み、既存キャッシュはサイズとSHA-256が一致する場合に再利用します。取得中は固定バッファで`.part`へ保存し、サイズ・ハッシュ検証後に確定名へatomic renameします。

再取得では転送前後のremote名、相対パス、サイズ、更新日時、hash type、hash valueを比較します。転送中にremoteが変化した場合は`verification_failed`として拒否し、既存の正常モデルを維持します。既存モデルを置き換える場合も一時バックアップを作成し、DB commit失敗時は旧ファイルへ復元します。

`rclone.conf`はGitへ保存せず、`RUNPOD_LORA_STUDIO_RCLONE_CONFIG_PATH`で指定します。起動時またはUIから、実行ファイル、設定ファイル、remote、接続、ローカルキャッシュ、転送一時領域を検証します。remote名と相対パスは`..`、絶対パス、制御文字、別remote指定を拒否します。

スナップショットは`completed`かつ再検証に成功したものだけを対象にします。既定の保存先は`<remote>:lora-studio/projects/<project_id>/snapshots/<snapshot_id>/`です。転送前にドライランを実行し、remote上の同名内容、manifest、snapshot `content_sha256`、設定、ポリシーを確認します。衝突ポリシーは`fail_if_exists`、`skip_identical`、`copy_missing`、`overwrite_changed`から選択でき、既定値は安全側の`skip_identical`です。remote hashが取得できない場合の既定方針は`error`で、古いmanifestのlocal SHA-256だけではskip_identicalと判定せず、衝突またはコピー扱いにします。限定的なremoteメタデータ比較を選ぶ場合も完全な内容検証ではありません。

転送後はverification levelに応じて各ファイルの存在・サイズ、remote hash、必須manifest、snapshot content hashを検証し、ローカルmanifestへ成功・失敗・スキップ件数、サイズ、ハッシュ、remoteメタデータ、rcloneバージョン、設定スナップショットを記録します。`remote_hash_and_size`はremote hashとサイズの一致、`manifest_metadata_and_size`はmanifestメタデータとサイズ、`existence_only`は存在のみ、`not_verified`は未検証、`verification_failed`は検証失敗を表します。古いtransfer-manifest.jsonのlocal SHA-256だけではremote実体の同一性を確認できないため、skip_identicalや転送後検証の成功には使用しません。進捗は完了済みファイルと現在ファイルを分けた累積値で、スキップは転送バイト数へ含めません。キャンセル、指数バックオフ、rclone子PID、worker ID、heartbeat、アプリ再起動後のstale検出をDBへ保存します。DBトランザクションは短く保ち、画像やモデル全体をbytesへ展開しません。

通常処理では`rclone sync`、remote側の無関係なファイル削除、危険な上書き、秘密情報のログ出力を行いません。Google Drive実通信の手動確認は、rclone設定を配置したRunPodで`rclone version`、`rclone listremotes`、接続確認、モデル一覧、ドライラン、少量のモデル取得、completedスナップショットの転送・再検証の順に実施してください。認証情報がない環境の自動テストはFakeStorageTransferAdapterを使用します。

## Phase 8A: Danbooru検索・取得候補・取得計画

「画像取得」タブでは、必須／除外タグ、rating、スコア、解像度、拡張子、候補数を検証してDanbooruの公開メタデータAPIを検索できます。検索結果はsource typeと外部post IDをキーにSQLiteへ保存し、既存画像・既存計画・URL・拡張子・rating・解像度をローカルで再確認します。除外理由は候補ごとに固定コードで保持し、UIでは日本語で表示します。

Phase 8Aは画像本体をダウンロードせず、外部画像を初期表示せず、確定時も不変の取得計画だけを保存します。API通信は固定HTTPSエンドポイント、応答サイズ上限、JSON schema確認、1ワーカーのレート制限、429等の限定的な指数バックオフ、秒数またはHTTP-date形式のRetry-After、待機中キャンセルを使用します。認証が必要な場合は`DANBOORU_LOGIN`と`DANBOORU_API_KEY`を環境変数へ設定します。これらはDB、ログ、UI、fingerprintへ保存しません。検索workerのclaimはDBの条件付き更新とclaim tokenで保護され、stale searchはcursorから安全に再claimされます。取得計画のexternal post reservationにより、同じpostを複数のplanへ同時確定しません。

検索ページは、APIへ渡した`request_cursor`を先に保存し、ページ内候補のupsertと保存完了を示す`current_cursor`／`page_count`のcheckpointを同一transactionで確定します。途中停止・claim喪失・DB commit失敗では未完了ページのnext cursorを進めず、stale再開時は保存済みrequest cursorから同じページを冪等に再実行します。cursor履歴はfingerprintでDBへ保存し、worker世代をまたぐcursor loopを検出します。候補数上限に到達した場合は未処理の残りページを再開対象にせず、`candidate_limit`として検索をcompletedにします。

検索結果の確認と計画確定の間にメタデータや重複状態が変わった場合は確定を拒否します。確定済み計画はfingerprintで冪等に再取得できます。

## Phase 8B: 画像本体の安全な取得・検証・登録

確定済み取得計画は、画像本体をUUID命名の`.part`へストリーミングし、許可済みHTTPSホスト、応答サイズ、Content-Range、MD5、画像実体、拡張子、寸法、ピクセル数を検証してからPhase 1の`ImageAsset`へ登録します。検証前のファイルは公開領域へ置かず、登録は原画像・PNGサムネイル・DB・外部post provenanceを二段階で確定します。既存SHA-256画像は再利用し、外部postのlinkだけを追加します。

取得workerはjob/item/attempt、claim token、heartbeat、キャンセル、stale復旧、指数バックオフ、Range再開をSQLiteへ保存します。UIにはURL、絶対パス、秘密情報を表示せず、manifestにも含めません。`FakeDownloadTransport`を使った破損、サイズ不一致、再開、同一SHA linkのテストを用意しています。Google Drive同期、完了manifestのDriveコピー、Pod Stop／TerminateはPhase 9の対象です。

取得開始時はplanの不変構造を一括検証し、worker実行時の外部post再確認はitem単位で行います。そのため、削除・metadata変更・不正MD5などの恒久的なsourceエラーが一部itemで発生しても、他のitemは処理を継続してpartial successとして記録します。source／HTTPの一時エラーだけを再試行し、401／403／404などの認証・権限・未検出エラーは再試行しません。attempt番号はjob itemごとに停止・stale復旧後も累積し、開始記録と監査更新はworker generationまで含むclaimで保護されます。

start_jobはDB上のconfirmed plan、search result、reservation、fingerprintなど不変構造だけを検証し、job作成前に外部postを一括取得しません。最新のsource存在・metadata・URL・MD5は各itemのworker処理直前に確認し、実行時に使わない最新URLをjob itemへ保存しません。

stale復旧ではdownloadingをpendingへ戻し、downloaded／validating／validatedの整合する`.part`をvalidation pendingとして検証から再開します。validation pendingは再ダウンロードせず、Pillow、サイズ、MD5、SHA-256、寸法、形式の検証から処理します。importingは既存のsource linkと検証済みファイルを確認して冪等に完了へ復旧し、対応するrunning attemptを終端化して不要な`.part`を清掃します。Range応答はContent-Range全体、Content-Length、期待サイズ、同種のETag／Last-Modified validatorを個別に検証します。

stale claimの回収は、job ID、running status、stale heartbeat、worker ID、claim token、worker generationを含む条件付き更新で原子的に行います。回収後のjobは`queued`へ戻し、新しいworker claimの下でcounter再計算、terminal status判定、manifest生成、`completed_at`設定、active key解除を行います。そのため、import済みitemだけが残るstale jobも、専用workerのclaim後にjobとmanifestを終端化できます。

source metadata取得の`get_post()`は単発requestとし、item attemptをretryの正本にします。request前後とRetry-After待機中はcancel、claim token、worker generation、heartbeatを確認します。cancelで呼出元workerが先に戻る場合も、実transportはクライアント単位の上限1 executorで継続し、source limiterのleaseと`after_request`は実transport終了後にだけ解放・実行します。そのため、cancel直後に次の外部requestを開始せず、background threadがrequestごとに無制限に増えません。429とRetry-After、callback／transport／cancel／claim喪失時の例外・release監査を維持し、URL、Authorization、API key、raw responseはログへ出しません。

manifestはworker generationとランダムUUID断片によるworker固有のtemporary／final fileへ書き込み、JSON書込みとSQLite transactionを分離します。`project_root`から`manifests`までの全親componentを`lstat`してsymlink、traversal、projects root外を拒否し、不足ディレクトリは検証しながら順に作成します。temporaryは`O_CREAT|O_EXCL`（対応環境では`O_NOFOLLOW`／directory fd）で作成し、file fsync、claim再確認、atomic replace、manifest directory fsync、親とfinalの再検証、claim条件付きDB UPDATE、commit後の参照path確認をこの順序で行います。claimを失ったold workerやDB更新に失敗したworkerは自分が作成したfileだけをcleanupし、commit結果が不明な場合にDBが参照済みならfileを削除しません。DBが参照するpathはprojects root内のregular fileに限定し、共有`.manifest.tmp`や固定`manifest.json`の競合を使用しません。plan検証の恒久エラー時は、`PENDING`、`DOWNLOADING`、`DOWNLOADED`、`VALIDATION_PENDING`、`VALIDATING`、`VALIDATED`、`IMPORTING`の全非終端itemとrunning attemptを同じfailure codeでFAILED・非retryableに終端化し、安全な`.part`をcleanupしてからcounterを再計算し、manifestを生成します。cleanup不能・不審pathはitemの`part_cleanup_warning`、manifest、`AcquisitionItemView`へ固定コード（`PART_CLEANUP_FAILED`、`PART_PATH_INVALID`、`PART_SYMLINK_REJECTED`、`PART_NOT_REGULAR_FILE`）で保存します。保存先はAlembic `0032_phase8b_part_cleanup_warnings`で追加し、絶対パスやraw exceptionは保存しません。

最新HEADでの品質チェックは、`ruff format --check .`、`ruff check .`、`mypy src`が成功し、`pytest`は`348 passed, 10 skipped, 77 warnings`です。
## Phase 6A: SDXL LoRA学習ジョブ基盤

Phase 6Aでは、完成済みデータセットスナップショットと検証済みローカルモデルを入力に、学習設定・ジョブをSQLiteへ保存し、安全な引数配列で `sdxl_train_network.py` を起動します。ジョブはPID、worker heartbeat、stdout/stderrログ、終了コードを記録し、キャンセル、stale復旧、boundedなログ末尾取得に対応します。

学習コマンドは許可されたtrainerと型付き追加オプションだけを受け付け、`shell=False`、許可ディレクトリ、固定された環境変数を使用します。Phase 6Aの範囲には成果物のGoogle Drive同期、epoch/loss解析、resume、TensorBoard、Pod停止・Terminateは含まれません。
### Phase 6Aレビュー対応の設定境界

- Python実行ファイルは`sys.executable`を基準にした検証済み環境から固定し、UI/APIで任意の実行ファイルを指定できない。
- trainerは`training_sd_scripts_root`内の固定スクリプトだけを実行し、`workspace_root`配下の任意スクリプトは実行しない。Python実行ファイルはresolve後の完全パスで信頼判定し、venv symlinkにも対応する。
- network moduleは`networks.lora`、optimizerとschedulerはsd-scriptsで確認済みの固定候補だけを許可する。
- repeatsはPhase 4のdataset TOMLにある各subsetの`num_repeats`を正とする。学習設定では重複指定せず、snapshotの元TOMLは変更しない。

## Phase 6B: 学習進捗解析と成果物追跡

Phase 6Bでは、workerのメモリ状態を正とせず、SQLiteに保存した進捗・metric・artifactをGradioへ復元表示します。stdout/stderrはbyte offsetから増分解析し、ANSI制御文字、不正UTF-8、carriage return、未完了行を安全に扱います。stdoutとstderrの未完了バイト列・parser stateは別々に保存し、片方のログをもう片方へ連結しません。

- 総stepはログ明示値を優先し、ない場合だけsnapshot由来のdataset TOMLと`num_repeats`から推定します。推定式は`ceil(sum(image_count * num_repeats) / (batch_size * gradient_accumulation_steps * world_size)) * epochs`です。
- loss履歴は同一job・metric・stepでupsertし、最大件数を超えた場合は決定的に間引きます。ETAは直近のstep速度から上限付きで計算し、終了jobでは表示しません。
- 学習実行時の`--output_dir`は`training/jobs/<job_id>/output`へ固定し、設定に保存された候補出力先を実行時の共有領域として使用しません。成果物はこのジョブ専用output配下のoutput name一致`.safetensors`と検証済みstate directoryだけを発見するため、異なるjobの同名ファイルも混ざりません。symlink、一時ファイル、サイズ上限超過、壊れたheaderは採用せず、ファイルを削除・移動しません。
- ログのinode変更またはtruncateを検出した場合はoffsetと保留中UTF-8バイト列を破棄して新しいファイルの先頭から読み直します。旧形式job（専用runtime/outputがないもの）は共有出力先を走査せず、警告として扱います。
- safetensorsはheader、tensor、metadata、サイズ・mtime、SHA-256を基本検証します。pickle系ローダーやGPUへのtensorロードは行いません。
- Phase 6Bの`succeeded`はプロセスのexit code 0を意味し、最終LoRAの品質保証やGoogle Drive同期を意味しません。stateからのresumeはPhase 6C、成果物同期と完了manifestはPhase 9の対象です。

## Phase 6C: SDXL LoRA学習stateからの安全な再開

Phase 6Cでは、`failed`、`canceled`、またはプロセス終了を安全に確認した`stale` jobに登録されたtraining stateを検証し、新しい子jobとして再開します。stateはjob専用output配下の通常ディレクトリだけを対象に、symlinkを拒否し、ファイルをストリームコピーしてSHA-256を再検証します。pickle、torch、YAMLなどのstate deserializeは行いません。

- Alembic `0013_phase6c_training_resume`で親子job、resume artifact、検証状態、初期epoch/stepと進捗オフセットを保存し、`0014_phase6c_resume_request_fingerprint`でtarget configを含む再開要求の一意性を、`0015_phase6c_state_position_provenance`で検証済み位置と出所を保証します。親jobのstatus、progress、metric、artifactは変更しません。
- dataset snapshot/content/TOML、モデル、trainer、LoRA構成、optimizer/scheduler、precision、cache、gradient checkpoint、resolution、batch、repeats、seed、sd-scripts root、信頼済みPython、command builder versionを再開前に比較します。epochの延長は許可しますが、縮小やmetadata欠落は拒否します。
- childの`runtime/resume/source-state`へatomicにコピーし、`config/resume-state-manifest.json`を作成してから、固定された`--resume`引数で起動します。開始直前にも同じ検証を再実行します。
- 選択したstate artifactのepoch/stepを初期値とoffsetの唯一の基準にします。artifact、state directory名、`training_state.json`、`state.json`、`resume-state-manifest.json`の値は厳格な非負整数として検証し、複数情報源の不一致、target epochsまたは推定total stepsを超える値、上限超過を拒否します。採用値と出所をpreview、DB、manifest、fingerprintへ保存します。親jobの最新progressはoffsetに使用せず、値が異なる場合はpreviewとmanifestへwarningを保存します。再開元の進捗・metricをchildへコピーせず、child outputだけをartifact走査します。

## Phase 7A: 実行環境診断と学習パラメータ推奨

Phase 7Aでは、CUDA/GPU/VRAM、bf16、xformers、bitsandbytes、sd-scriptsの実行環境を診断し、完了済みdataset snapshotの統計を入力として決定論的なLoRA学習設定を提示します。診断結果と推奨結果はSQLiteへスナップショットとして保存し、後から同じ入力 fingerprintを検証できます。

- 推奨エンジンはルールベースで、concept type、quality/speed profile、実効画像数、解像度、VRAM安全マージンからbatch、dim/alpha、epoch、optimizer、scheduler、precision、cache/checkpointingを決めます。gradient accumulationはPhase 7Aでは1に固定します。
- VRAM見積もりはtotal/free VRAMを区別し、安全マージンを差し引いて判定します。GPU未検出、bf16非対応、依存不足、空caption、重複、未確認類似グループなどは警告として表示し、blocking警告がある推奨は適用できません。
- 「推奨設定を適用」はDBから推奨とrequestを再読込し、関連付け、入力fingerprint、snapshot/modelの状態、現在の診断、ユーザー編集後の危険度を再検証してから、既存の`TrainingConfigInput`検証を通して保存します。blocking warningがある推奨はUIだけでなくサービス側でも拒否します。free VRAMは安定fingerprintには含めず、適用直前に再診断します。手動設定は推奨provenanceを持たずに保存できます。dataset snapshotのrepeatsは変更せず、推奨ID、エンジンバージョン、各項目の`recommended`/`applied`差分を設定へ記録します。UI stateには推奨本体を保持せず、IDとfingerprintだけを保持します。Phase 7Bの自動探索や学習中適応は対象外です。

## Phase 7B: 学習実績による推奨補正と履歴比較

学習開始時には推奨結果とは独立して、ジョブ専用の実行環境snapshotを一度だけ保存します。snapshotには論理GPU・物理GPU・GPU UUID fingerprint、architecture/compute capability、total VRAM、CUDA利用可否、正規化済み`CUDA_VISIBLE_DEVICES`、sd-scripts/xformers、検出version、検出時刻だけを記録し、秘密情報やホスト情報は保存しません。手動設定・旧設定・推奨snapshot欠落でも同じ経路で記録します。

- GPU選択は実行時snapshotを基準にし、`CUDA_VISIBLE_DEVICES`の論理番号とnvidia-smiの物理番号/UUIDを対応付けます。複数GPUでは対象PIDのprocess identity/groupとUUIDが一意に検証できた場合だけ実測GPUを採用し、判定不能時に先頭GPUへフォールバックしません。
- 推奨時snapshotは比較用 provenance に限定します。実行GPUが変わった場合は警告・校正除外またはジョブ失敗コードを保存し、実行時snapshotと測定失敗コード（`GPU_IDENTITY_UNAVAILABLE`、`EXPECTED_GPU_NOT_FOUND`、`TARGET_PID_NOT_FOUND`、`PROCESS_IDENTITY_MISMATCH`、`PROCESS_GROUP_MISMATCH`、`AMBIGUOUS_GPU_SELECTION`、`GPU_CHANGED_DURING_JOB`）を保持します。

Phase 7Bは、完了した学習jobから速度、VRAM使用量、終了理由を収集し、Phase 7Aのルール推奨を補足する決定論的な校正機能です。過去実績だけで設定を決定せず、GPU fingerprint、VRAMクラス、アーキテクチャ、解像度、batch、gradient accumulation、effective batch、LoRA module/dim/alpha、precision、optimizer、cache/checkpointing、world size、sd-scripts version、xformers可否が一致する履歴だけを利用します。

- 成果は`training_execution_summaries`へジョブ単位で冪等に保存し、OOM・キャンセル・不明失敗は速度校正から除外します。ログ本文や秘密情報は保存せず、許可された証拠コードだけを失敗分類へ記録します。
- 学習中はheartbeat、進捗、artifact走査とは独立して、既定5秒間隔で`nvidia-smi`を測定します。固定queryでジョブPIDのcompute行だけを対象プロセスとして検証し、PID identityまたはGPU UUIDを確認できない場合はtarget peakを保存せず、全GPUのfree/minimumだけを低信頼で保持します。保存するのは短縮GPU UUID、合計VRAM、開始前・最小free・終了後、target/全GPU/他プロセスpeak、件数・失敗件数・時刻範囲・測定versionです。
- 校正は`recommendation_calibration_snapshots`と元サマリーの関連テーブルへ保存します。サンプル数、成功数、OOM数、中央値・保守的percentile、信頼度、source fingerprintを表示し、履歴が変化したスナップショットはstaleとして扱います。
- メモリ校正はtarget peak、2点以上のサンプル、identity検証、正のcoverage、他プロセス影響50%以下を必須とします。OOMのbatch低下はGPU/VRAMクラス、解像度、batch、gradient accumulation、optimizer、precision、cache/checkpointing、LoRA module/dimが一致するmedium以上の履歴だけで提案し、異なるdimやbatchの履歴では増加も低下も行いません。
- summary内容fingerprintと校正状態fingerprintを分離し、include/exclude、理由、再分類、force recollect、メモリ集約更新は関連snapshotだけをstaleにします。適用直前にもsnapshotのstaleと推奨設定・現在GPUの互換性を再検証します。校正の再構築や再収集だけで学習を開始することはありません。UIから履歴の更新、performance再収集、校正再構築を実行できます。
- `failed`、`canceled`、`stale`のいずれでもPID、process group、process identity、worker情報、heartbeatを確認し、終了を確証できないjobは再開しません。プロセスのkillは行いません。
