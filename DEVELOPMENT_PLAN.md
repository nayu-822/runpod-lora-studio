# RunPod LoRA Studio 開発計画書

- 文書版数: 1.0
- 作成日: 2026-07-19
- 対象リポジトリ名（推奨）: `runpod-lora-studio`
- 関連文書: `RUNPOD_SDXL_LORA_TOOL_DESIGN_SPEC.md`, `CODING_RULES.md`, `AGENTS.md`

## 1. 目的

本計画書は、設計仕様書に基づいて、RunPod上で稼働する自分専用のSDXL LoRA作成ツールを段階的に実装するための開発順序、成果物、完了条件を定義する。

ツール本体はGitで管理し、RunPod上でGradioを起動する。学習元モデル、学習に使用した画像、タグ付け後テキスト、完成LoRA、設定、ログはGoogle Driveへ保存する。学習途中のキャッシュ、state、途中チェックポイントはRunPod内に保持し、正常完了後に必要な成果物だけをGoogle Driveへ同期する。

## 2. 開発原則

1. 最初にローカル画像からLoRA作成まで完走する最小経路を作る。
2. Danbooru取得、品質判定、高度な推奨機能は、その後に追加する。
3. 原画像・タグ・学習設定に対して非破壊処理を行う。
4. 長時間処理はGradioイベント内で同期実行せず、別プロセスとして管理する。
5. Google Drive同期成功前にPodをTerminateしない。
6. 自動除外より、除外候補の提示と手動確認を優先する。
7. 外部ライブラリや学習バックエンドのバージョンを固定し、再現性を確保する。
8. 各フェーズでテスト可能な状態を保ち、大規模な一括実装を避ける。

## 3. 想定リポジトリ構成

```text
runpod-lora-studio/
├── AGENTS.md
├── CODING_RULES.md
├── DEVELOPMENT_PLAN.md
├── README.md
├── pyproject.toml
├── requirements.lock
├── .env.example
├── .gitignore
├── scripts/
│   ├── bootstrap_runpod.sh
│   ├── start.sh
│   └── verify_environment.py
├── src/
│   └── runpod_lora_studio/
│       ├── app.py
│       ├── config/
│       ├── domain/
│       ├── services/
│       │   ├── acquisition/
│       │   ├── deduplication/
│       │   ├── quality/
│       │   ├── tagging/
│       │   ├── dataset/
│       │   ├── training/
│       │   ├── storage/
│       │   └── runpod/
│       ├── adapters/
│       │   ├── sources/
│       │   ├── taggers/
│       │   ├── trainers/
│       │   └── storage/
│       ├── ui/
│       ├── jobs/
│       └── persistence/
└── tests/
    ├── unit/
    ├── integration/
    └── fixtures/
```

## 4. フェーズ別計画

## Phase 0: リポジトリ基盤

### 実装内容

- Pythonプロジェクト初期化
- `src`レイアウト採用
- `pyproject.toml`作成
- Ruff、mypy、pytestの導入
- `.env.example`作成
- `.gitignore`作成
- 構造化ログの基盤
- 設定読込基盤
- RunPod環境確認スクリプト
- Gradioの最小画面
- CIでlint・型チェック・テストを実行

### 完了条件

- `python -m runpod_lora_studio.app`でGradioが起動する
- `ruff check .`が成功する
- `mypy src`が成功する
- `pytest`が成功する
- 秘密情報や成果物がGit管理対象外になっている

## Phase 1: プロジェクト管理とローカル画像登録

### 実装内容

- プロジェクト作成・一覧・選択
- SQLiteによるプロジェクトメタデータ保存
- ローカル画像アップロード
- 原画像のSHA-256計算
- 画像メタデータ保存
- サムネイル生成
- 採用・保留・除外状態
- 原画像を削除しない非破壊管理

### 完了条件

- 新規プロジェクトを作成できる
- 複数画像を登録できる
- 再起動後に画像一覧と選択状態を復元できる
- 採用・保留・除外を切り替えられる

## Phase 2: 重複判定と基本品質検査

### 実装内容

- SHA-256による完全重複検出
- pHashによる近似重複候補検出
- 画像破損チェック
- 最低解像度チェック
- 極端な縦横比チェック
- 単色・低情報量画像の候補判定
- ぼけの参考スコア
- 除外理由の複数保持
- 類似グループ比較UI

### 完了条件

- 完全重複が自動検出される
- 近似重複が候補としてグループ化される
- 自動判定をユーザーが上書きできる
- 自動除外結果から画像を復元できる

## Phase 3: 自動タグ付けと手動タグ選別

### 実装内容

- WD系Taggerのアダプター
- 採用画像への一括タグ付け
- タガーモデル・閾値の保存
- タグ出現回数の集計
- 出現画像数の多い順でタグ一覧を表示
- 各タグに「残す」チェックボックスを表示
- チェックを外したタグを全対象キャプションから削除
- 出現回数、出現率、カテゴリを表示
- タグ名検索・絞り込み
- 全選択・全解除
- 指定出現率以上のみ選択などの補助操作
- 削除適用前のプレビュー
- 元タグと最終タグを分離保存
- トリガーワードを先頭へ一括付与

### チェックボックスの意味

- チェックあり: 最終キャプションに残す
- チェックなし: 最終キャプションから削除する
- 初期値: 原則としてチェックあり
- 自動削除は行わず、ユーザー確定後に反映する

### 完了条件

- 採用画像からタグ頻度一覧を生成できる
- タグが出現回数降順で表示される
- チェック操作に基づいて全キャプションを更新できる
- 適用前後の差分を確認できる
- 再実行しても元の自動タグを失わない

## Phase 4: データセット生成

### 実装内容

- 採用画像と最終キャプションの対応検証
- データセットスナップショット
- SDXL用dataset TOML生成
- 解像度分布・縦横比分布
- タグ頻度レポート
- 重複率・類似率
- トリガーワード付与率
- キャプション空欄チェック
- 学習前警告

### 完了条件

- 学習に使う画像とテキストが固定版として保存される
- 過去の学習Runが後の編集で変化しない
- 学習開始前の必須検査が実行される

## Phase 5: モデル管理とGoogle Drive連携

### 実装内容

- rclone設定存在確認
- Google Drive上のモデル一覧取得
- 指定モデルをRunPod作業領域へ`rclone copy`
- ダウンロード済みモデルの再利用
- モデルSHA-256記録
- Google Drive保存パス設定
- 同期ドライラン
- 転送失敗時の再試行
- 同期結果マニフェスト

### 保存方針

Google Driveへ保存するもの:

- 学習元モデル
- 採用した学習画像
- タグ付け後の最終テキスト
- データセット設定
- 学習設定
- 完成LoRA
- 完成時のサンプル
- 学習ログ
- 完了マニフェスト

RunPod内だけに一時保存するもの:

- latent cache
- text encoder cache
- 一時変換画像
- 学習途中state
- 途中epochのLoRA
- 一時ログ
- ダウンロード途中ファイル

### 完了条件

- Google Driveからモデルを取得できる
- 学習用画像・テキスト・成果物の保存先を設定できる
- `rclone copy`の失敗を検知できる
- `rclone sync`を通常処理で使用しない

Phase 5のレビュー対応を完了し、完了扱いへ更新する。Alembic 0007でモデル、モデル転送、汎用ストレージ転送ジョブ、転送項目、プロジェクト別保存設定を追加し、0008でworker heartbeatとrclone子PID、0009で累積転送進捗を追加した。rcloneは引数配列・`shell=False`で実行し、設定ファイルをログへ出力しない。モデル取得は保存済みSHA-256、remote識別情報、転送前後remote情報を検証し、安全な置換・ロールバックと指数バックオフを行う。

- completedスナップショットのみを再検証してGoogle Driveへ転送する
- 転送前ドライランと`fail_if_exists`／`skip_identical`／`copy_missing`／`overwrite_changed`の衝突規則を提供する
- 転送後にファイルサイズ、manifest、設定、snapshot content hashを検証する
- `skip_identical`はremoteの比較可能なハッシュとサイズが一致した場合だけ適用し、古いmanifestのlocal SHA-256だけではスキップしない
- 成功・失敗・スキップ件数、完了済み／現在ファイルの累積進捗、キャンセル要求、stale状態をDBへ保存する。スキップは転送バイト数へ含めない
- rclone子PID、worker ID、Future状態をheartbeatと併せて確認し、実行中のrcloneをstaleにしない。workerは進捗出力とは独立した定期heartbeatを送る
- 通常処理で`rclone sync`を使わず、remoteの無関係なファイルを削除しない
- rclone.conf、remote名、相対パス、モデル拡張子、サイズ上限を検証する

### Phase 5の完了条件

- Google Driveから許可されたモデルを一覧・検索し、ローカルへ安全に取得・再利用・再検証できる
- completedスナップショットをドライラン、衝突判定、転送、manifest検証付きで保存できる
- 転送をキャンセル・再試行でき、再起動後に実行中プロセスの不在をstaleとして検出できる
- 0009を空DB、0006適用済みDB、0007／0008適用済みDBへ適用でき、既存モデル・スナップショット・転送ジョブを変更しない

0009で追加した`completed_transferred_bytes`と`current_file_transferred_bytes`は既存runningジョブでは0から開始する。既存のheartbeat／PIDは変更せず、アプリ再起動時のstale判定に従って回復する。転送検証は、設定値ではなくitemの実績からmanifest全体のverification levelを決定し、remote hash fallbackの実績もitemへ保存する。

## Phase 6: SDXL LoRA学習実行

### 実装内容

- sd-scriptsの呼び出し
- TOMLから引数配列を生成
- `subprocess.Popen`による別プロセス起動
- PID・Run ID管理
- 標準出力と標準エラーの保存
- epoch、step、lossの解析
- 停止操作
- 異常終了の検知
- 学習stateからの再開
- Gradio再接続後の状態復元

### 完了条件

- Gradioから学習を開始できる
- ブラウザを閉じても学習が継続する
- ログと進捗を再接続後に確認できる
- 停止・失敗・正常終了を区別できる

## Phase 7: 推奨パラメータ

### 実装内容

- GPU名・VRAM・bf16対応の取得
- 画像枚数・実質画像枚数の算出
- 概念種別別プリセット
- batch size、dim、alpha、epoch、repeatsの提案
- 推定総step表示
- 推奨理由表示
- 危険設定の警告
- 手動設定との切替
- 推奨エンジンのバージョン保存

### 完了条件

- 推奨値が自動入力される
- 各推奨値の理由を確認できる
- 手動変更できる
- 設定不整合を学習前に警告できる

## Phase 8: Danbooru画像取得

### 実装内容

- `ImageSourceAdapter`定義
- Danbooruアダプター実装
- タグ、除外タグ、rating、最低スコア、最低解像度
- APIレート制限
- 429時のバックオフ
- 途中停止・再開
- 投稿IDと取得元URLの保存
- 既取得投稿のスキップ
- 破損・取得失敗の記録

### 完了条件

- Danbooru検索結果から指定件数を取得できる
- 再実行時に重複取得しない
- API制限時に安全に待機・再試行する
- 将来別取得元を追加できる構造になっている

## Phase 9: 完了処理・Google Drive同期・Pod終了

### 実装内容

- 学習正常終了の確認
- 完成LoRAの存在・サイズ・SHA-256確認
- 採用画像・最終テキスト・設定・ログの同期
- `rclone copy`完了確認
- 完了マニフェストの同期
- 同期後のGoogle Drive側確認
- 自動終了設定
- 何もしない / Stop / Terminateの選択
- RunPod REST API連携
- 終了前猶予時間
- 失敗時はPodを維持

### Terminate条件

以下をすべて満たした場合のみTerminate可能とする。

1. ユーザーが自動Terminateを明示的に有効化している
2. 学習プロセスが終了コード0で完了している
3. 完成LoRAの検証に成功している
4. Google Driveへの必要ファイル同期に成功している
5. 完了マニフェストをGoogle Driveへ保存できている
6. RunPod APIキーとPod IDを取得できている

### 完了条件

- 正常完了時にGoogle Driveへ成果物が揃う
- 同期失敗時はTerminateされない
- 自動終了の実行結果がログに残る

## Phase 10: サンプル評価と品質改善

### 実装内容

- 固定seedによるepochサンプル
- base modelとの比較
- LoRA強度別比較
- 評価プロンプトセット
- お気に入りepoch指定
- モデルカード生成
- プロジェクトエクスポート
- CLIP類似度等の補助評価

### 完了条件

- 同条件でepoch間を比較できる
- 最終採用LoRAを選択できる
- モデルカードと設定を一緒に出力できる

## 5. マイルストーン

| マイルストーン | 対象Phase | 到達状態 |
|---|---:|---|
| M1 基盤起動 | 0 | RunPod上でGradioが起動する |
| M2 データセット編集 | 1～4 | 画像選別・タグ選別・データセット固定ができる |
| M3 手動学習完成 | 5～6 | 指定モデルでLoRA学習を完走できる |
| M4 初心者支援 | 7 | 推奨パラメータで学習できる |
| M5 自動収集 | 8 | Danbooruからデータセットを作れる |
| M6 自動保存・終了 | 9 | GDrive同期後にPodを安全に終了できる |
| M7 評価機能 | 10 | epoch比較と最終成果物出力ができる |

## 6. Codexへの依頼単位

Codexへは、原則として1依頼につき1フェーズまたは1つの明確な機能単位で依頼する。

良い依頼例:

- Phase 0のプロジェクト基盤だけを実装する
- タグ頻度一覧と保持チェックボックスだけを実装する
- rcloneによる成果物コピーと検証だけを実装する
- 学習プロセスの開始・停止・状態復元だけを実装する

避ける依頼例:

- 仕様書全体を一度に完成させる
- UI・DB・学習・同期を同時に全面改修する
- テストなしで多数の機能を追加する

## 7. 各対応後にCodexが報告する内容

Codexは実装・修正後、必ず以下を報告する。

1. 実装概要
2. 変更ファイル一覧
3. 主な設計判断
4. 実行したテスト・静的解析
5. テスト結果
6. 未解決事項または注意点
7. 動作確認手順
8. **推奨Gitコミットメッセージ**

コミットメッセージはConventional Commits形式を基本とする。
typeとscopeは英語の固定語を使用し、subjectと本文は日本語で記述する。

例:

```text
feat(tagging): 出現回数順のタグ保持チェック機能を追加する
```

必要に応じて本文も提示する。

```text
feat(storage): 完成した学習成果物をGoogle Driveへ同期する

- 完成LoRA、キャプション、画像、設定、ログをrcloneでコピーする
- Runを完了扱いにする前にコピー済み成果物を検証する
- 同期に失敗した場合はPodを自動Terminateしない
```

## 8. Definition of Done

各タスクは、以下を満たして完了とする。

- 要件を満たすコードが実装されている
- 既存機能を破壊していない
- 必要なテストが追加されている
- lint、型チェック、テストが成功している
- エラー時にユーザーが原因を理解できる
- 秘密情報をログへ出力していない
- READMEまたは関連文書が必要に応じて更新されている
- Codexが推奨コミットメッセージを提示している
## Phase 2A 完了

画像検査基盤、SHA-256完全重複検出、最低解像度・極端な縦横比・低情報量・ぼけの基本検査を実装済み。
近似重複（pHash）と類似画像比較はPhase 2Bで実装する。
Phase 2Aの一括検査は数千枚程度を想定し、ストリーミング化はPhase 2B開始前の改善課題とする。
## Phase 2B 実装状況: pHash近似重複検出・類似グループ比較UI

Phase 2Bを実装済みとする。ImageHash/PillowによるEXIF補正済みpHash計算、画像単位の失敗保存、ハミング距離の連結成分グループ化、代表候補と手動代表、正規化した手動否定ペア、バッチ処理、Alembic 0004、Gradio比較UIを追加した。自動処理は候補提示に限定し、原画像・Phase 2A検査結果・SelectionStateを自動変更しない。

### Phase 2完了条件

Phase 2AのSHA-256／基本品質検査と、Phase 2BのpHash近似重複検査・比較UIを合わせてPhase 2の完了条件を満たす。CLIPによる意味的類似判定、数万枚規模向けのBK-tree/LSH、学習処理・外部同期は後続Phaseの対象とする。

## Phase 3 実装状況: 自動タグ付け・手動タグ選別・最終キャプション編集

Phase 3のレビュー対応を完了した。採用画像を対象に、FakeTaggerでテスト可能なTaggerAdapter境界、WD Tagger互換のモデル識別・環境検証、画像単位の結果保存、失敗継続、キャンセル・stale復旧、タグ頻度集計、保持ルールの草稿・プレビュー適用、手動キャプションのポリシー・トリガーワード・履歴復元、Gradio UI、Alembic 0005を追加済みである。

生成キャプションはsource tagsとfinal captionを分離し、手動編集や復元を履歴化する。プレビューの署名検証、ルール保存とキャプション更新のトランザクション、採用画像限定、モデル設定スナップショットを完了条件とする。WDモデルはRunPod環境へ配置するか、明示的に許可した場合だけ一時領域へ取得し、通常のテストでは外部ダウンロードを実行しない。

レビュー対応では、completed/partially_failed Runの成功結果だけを利用し、失敗結果を頻度・キャプションの分母から除外する。プレビューはaccepted画像集合、各SelectionState、current caption revision、元タグ結果、ルール、トリガー、方針、変更前後を署名し、適用前に再検証する。TaggerRunのtarget_image_countは実際に推論する画像数、skipped_image_countは既存結果などで対象外になった画像数とする。

Run開始時点の対象画像ID・パスは現在メモリへ保持するが、画像本体は一括読込しない。数千枚程度を想定し、将来はRun対象画像テーブルまたはワーカー側keyset paginationへ移行する。Run開始後のSelectionState変更は実行対象集合には反映せず、プレビュー適用時は変更を検知して拒否する。

### Phase 3の完了条件（レビュー対応後に再確認）

- 採用画像へ自動タグを付け、失敗・キャンセル・再実行状態を復元できる
- タグ頻度と保持ルールをプレビューし、最終キャプションを安全に適用・復元できる
- SQLiteへTaggerRun、検出タグ、キャプション、ルール、編集履歴を保存できる
- Phase 4のデータセットTOML、学習、成果物同期は後続として境界が保たれている

## Phase 4実装状況: 学習用データセット生成・固定スナップショット・学習前検査

Phase 4のレビュー対応を完了し、完了扱いへ戻す。Alembic 0006で、採用画像とcurrentキャプションを独立した不変スナップショットへ保存する機能、SDXL dataset TOML、manifest、設定を含む内容ハッシュ、容量検査、ハッシュ付きレポート、作成ジョブ、キャンセル、stale復旧、DB確定失敗からのmanifest回復、再検証を追加した。

- 対象は作成時点で`accepted`の画像だけで、currentキャプション・原画像の存在、可読性、DBハッシュ、ファイルサイズを必須検査する
- 品質・完全重複・pHash近似重複・トリガーワード不足は警告として表示し、確認なしでは作成しないが自動除外しない
- プレビュー署名には対象画像集合、SelectionState、原画像ハッシュ／サイズ、キャプションID／revision／本文ハッシュ、元TaggerRun、設定、トリガーワード、生成器バージョンを含める
- 作成中は一時ディレクトリへ画像を1枚ずつストリーミングコピーし、コピー後ハッシュ検証、TOML再パース、manifest・レポート生成の完了後にatomic renameする
- rename後のDB保存・commit失敗は`db_finalization_pending`へ記録し、確定ファイルを残してmanifestからitem/issueを冪等に回復できる
- `manifest_sha256`、`dataset_toml_sha256`、相対パス・画像／キャプション・TOML・設定を含む`content_sha256`を保存し、再検証失敗時はファイルを削除せず`corrupted`へ遷移する
- 画像、キャプション、manifest/report/CSV/TOML、安全マージンを合算し、filesystem空き容量が必要量未満ならDB作成前に拒否する
- 類似グループのreview status、member count、代表、否定済みペアを対象グループ単位で集計し、未確認・否定済みを自動除外せずレポートする
- キャプションはUTF-8 BOMなし、LF、末尾改行1つへ正規化し、TOMLの既定値はresolution 1024、bucket 256～2048／step 64、repeats 1とする
- 画像本体は一括ロードしない。現在はID・パス・キャプションメタデータをプレビュー中にメモリ保持するため想定数千枚までとし、将来は対象テーブルとkeyset paginationでストリーミング化する
- Phase 4は学習実行、Google Drive同期、スナップショット削除を行わず、Phase 5/6の入力境界を固定する

### Phase 4の完了条件

- 学習に使う画像とcurrentキャプションを固定版として保存できる
- プレビュー後の対象集合、SelectionState、原画像、キャプション、設定変更を適用前に検知できる
- 必須エラー、警告確認、コピー後検証、TOML検証、manifest検証が完了条件へ含まれる
- DBとファイルの再検証、破損状態、作成中ジョブのstale復旧を扱える
- rename後DB失敗を回復でき、completedスナップショットへ回復処理の影響を与えない
- content hashが相対パス、TOML、正規化設定の変更を識別できる
- 容量不足、未確認類似グループ、解像度・縦横比分布をプレビューとレポートへ反映できる
- SQLite/Alembic 0006、Gradio UI、単体テスト、品質コマンドを確認済みである
## Phase 6A レビュー対応中: SDXL LoRA学習ジョブ基盤とプロセス管理

- 学習設定・学習ジョブ、状態遷移、PID、worker heartbeat、終了結果をSQLiteへ保存
- `sdxl_train_network.py` の許可リストと型付き追加オプションによる安全なコマンド構築
- `Popen` worker、stdout/stderrログ、キャンセル、staleジョブ復旧、boundedログ表示を実装
- 完成済みsnapshot、manifest/content hash、検証済みローカルモデル、完了済み転送を開始条件として検証
- 最小限のGradio UIとFake process adapterを追加

成果物のGoogle Drive同期、epoch/loss解析、resume、TensorBoard、Pod自動停止・TerminateはPhase 6Aの対象外とする。
- Python実行ファイルはアプリ設定の検証済み実行環境に固定し、`/bin/sh`などの任意実行ファイルを受け付けない
- trainerは`training_sd_scripts_root`内の固定スクリプトだけを実行し、`workspace_root`配下の任意スクリプトは実行しない。Pythonはresolve後の完全パスで信頼判定し、venv symlinkにも対応する
- network module、optimizer、schedulerは固定許可リストからのみ選択する
- repeatsはPhase 4のdataset TOMLにある`num_repeats`を正とし、学習設定から重複指定しない

## Phase 6B完了: SDXL LoRA学習進捗解析と成果物追跡

- Alembic `0012_phase6b_progress_artifacts`で、training progress、metric history、artifact trackingを追加
- stdout/stderrの増分reader、offset・rotation・不完全UTF-8・不完全行の復元、sd-scripts/tqdm形式のparserを実装。2ストリームのremainderとparser stateは個別に永続化し、集約時は順序に依存しない
- ログ明示step優先、dataset TOMLの`num_repeats`による総step推定、進捗率、平滑化ETA、決定的metric間引きを実装
- jobごとの`training/jobs/<job_id>/output`を実行時出力先およびartifact走査先とし、output nameと許可拡張子に基づくartifact発見、state構造検証、safetensors header/metadata/SHA-256基本検証を実装。専用出力先のない旧形式jobは共有ディレクトリを走査しない
- DBから進捗・metric・artifactを再取得できるGradio表示と再解析・再走査操作を追加
- Phase 6Bの`succeeded`はexit code 0のプロセス結果を維持し、成果物の最終品質保証は行わない

## Phase 6C完了: SDXL LoRA学習stateからの安全な再開

- Alembic `0013_phase6c_training_resume`で、親jobとの関係、再開対象artifact、検証結果、初期epoch/step、local/cumulative進捗オフセットを追加し、`0014_phase6c_resume_request_fingerprint`でtarget configを含む再開要求fingerprintのunique制約、`0015_phase6c_state_position_provenance`で検証済みstate位置と出所を追加した
- `failed`、`canceled`、安全に再確認できた`stale`だけを再開元として許可し、active job、symlink、job output外、空state、容量・深さ・ファイル数超過を拒否する。state内容のdeserializeは行わない
- stateの安定性をストリームコピーとSHA-256で検証し、childの`runtime/resume/source-state`と`resume-state-manifest.json`へ保存する。親jobのstatus、progress、metric、artifactは変更しない
- dataset/model/trainer/LoRA/optimizer/scheduler/precision/cache/gradient checkpoint/resolution/batch/repeats/seed/runtimeの互換性をfingerprintで比較し、epochの縮小とmetadata欠落を拒否する。preview signatureにより再検証前の変更も検出する
- childには固定された`--resume`だけを追加し、開始直前にstateを再検証する。選択state artifact、directory名、検証済みmetadataのepoch/stepを厳格な非負整数として比較し、不一致やtarget epochs・推定total steps・設定上限超過を拒否する。採用値と出所をpreview・manifest・DB・fingerprintへ保存し、そのepoch/stepをoffsetの基準とする。親progressとの差異はwarningとして保存する。child outputだけを走査する
- `failed`、`canceled`、`stale`の全再開可能statusについてPID、process group、process identity、worker情報、heartbeatを確認し、終了を確証できないjobは再開しない。再開サービスからkillは実行しない
- 再開元選択、state選択、preview、child job作成・開始のGradio操作と、fingerprint、互換性、親子分離、コピー検証、active/stale境界のテストを追加した

Google Drive同期と完了manifestはPhase 9で実装する。Phase 6全体は、Phase 6Cまで完了したが、Phase 9連携を含む完成条件は未完了とする。

## Phase 7A完了: 実行環境診断と学習パラメータ推奨エンジン基盤

- `ComputeEnvironmentService`と`TrainingEnvironmentService`でGPU/CUDA/VRAM、bf16、依存ライブラリ、sd-scripts実行環境を診断し、診断結果を不変スナップショットとして保存する
- 完了済みdataset snapshotから、画像数、repeatsを反映した実効画像数、caption、trigger coverage、重複・類似グループ、解像度・aspect・bucket統計を計算する
- `RuleBasedRecommendationEngine`と`TrainingMemoryEstimator`で、許可済みoptimizer/scheduler/network moduleの範囲内から1件の決定論的推奨を生成する。GPUがない場合や安全VRAMを超える場合はblocking warningを付ける
- 推奨の適用は`RecommendationApplicationService`へ一元化し、DBからrequest/recommendationを再読込して関連付け、入力fingerprint、snapshot/model、現在の診断、ユーザー編集後のblocking riskを再検証してから既存の`TrainingService`のconfig検証を通して保存する。学習開始とは分離し、manual configはprovenanceなしで保存する。dataset snapshotは変更せず、推奨ID・engine version・各項目の`recommended`/`applied`差分をtraining configへ記録する。UI stateには推奨本体を保持せず、推奨ID・request ID・fingerprint・警告要約だけを保持する
- Alembic `0016_phase7a_recommendation_snapshots`、`0017_phase7a_recommendation_metadata`、`0018_phase7a_recommendation_input_config`で環境・推奨・適用 provenanceと再検証用の入力設定を保存する。free VRAMは安定fingerprintから除外し、適用直前に再診断する。Optuna等の自動探索、学習中適応、Phase 7B機能は対象外とする

## Phase 7B実装状況: 学習実績を利用した推奨補正・履歴比較・OOMフィードバック

- Torch indexはlogical indexとして扱い、physical GPU情報は固定nvidia-smi inventoryとUUID照合で確定する。mem_get_info()は(free, total)の順で取得し、free VRAMの変動をtotal VRAMへ混入させない。
- 開始前に実行GPUを確定できない推奨付きジョブは開始せず、PID queryで単一GPUを確定できた場合はimmutableな開始前snapshotとは別にruntime GPU identityを保存する。レビュー対応の品質確認が完了するまでPhase 7Bを完全完了扱いにしない。
- selected GPU identityは最初の確定値を保持し、後続の異なるUUIDは監査情報とGPU_CHANGED_DURING_JOBとして記録する。変更ジョブは速度・VRAM校正へ利用せず、関連calibrationをstale化する。

- 各TrainingJobの開始境界で、推奨の有無に依存しない不変の実行環境snapshotを追加する。論理/物理GPU、UUID fingerprint、architecture/compute capability、total VRAM、CUDA、正規化済みCUDA_VISIBLE_DEVICES、sd-scripts/xformers、detector version、detected_atを保存し、秘密情報やホスト情報は保存しない。
- GPU UUIDとCUDA_VISIBLE_DEVICESの対応、PID/process identity/groupを検証し、複数GPUやGPU変更時は安全側に倒す。実行時snapshotをsummary/calibrationの正本とし、推奨時snapshotは比較・stale判定だけに使う。手動jobでもtarget process peakを利用可能にする。
- Alembic `0022_phase7b_job_environment`と`0023_phase7b_memory_failure_codes`を追加する。既存の`0019`〜`0021`は変更しない。固定warning/failure codeとmigration後の再解析でも不変なGPU選択結果を保存する。

- `TrainingPerformanceCollector`でジョブ、進捗、metric、設定、環境snapshot、終端ログの限定末尾から、ジョブ単位の`TrainingExecutionSummary`を冪等に収集する。収集失敗は元jobの状態を変更しない
- `TrainingFailureClassifier`はOOM、system OOM、disk、model/dataset、dependency、cancel、stale、process killed、unknownを証拠コード付きで分類する。ログ本文、秘密、絶対パス、raw nvidia-smi出力は保存しない
- `NvidiaSmiGpuMemoryAdapter`は学習中に既定5秒間隔でheartbeat等と独立して動作し、固定argv、`shell=False`、timeout、出力上限、MiBからbytesへの検証済み変換を行う。正のjob PID、callerで検証済みのprocess identity/group、GPU UUIDを使ってtarget processを同定し、確認できないtarget peakはNULL、全GPUfreeは低信頼として保存する。測定は`training_memory_aggregates`へ1 job 1行の冪等集約として復元可能に保存する
- `RecommendationCalibrationService`はGPU/VRAMクラス/architecture、resolution、batch、gradient accumulation/effective batch、LoRA module/dim/alpha、optimizer、precision、cache/checkpointing、world size、sd-scripts version、xformersが一致する履歴だけを選択し、outlier除外、中央値、保守的percentile、信頼度、source fingerprintを決定論的に保存する。低信頼・stale・履歴不足時はPhase 7A baselineへフォールバックする
- `CalibratedRecommendationService`は時間・VRAMの比較とOOM後のbatch低下提案を提供するが、baselineの安全制約を緩和せず、推奨適用や学習開始を自動実行しない。UIから履歴再収集・校正再構築・履歴表示を操作できる
- Alembic `0019_phase7b_training_performance`を変更せず、`0020_phase7b_memory_measurements`でメモリ集約・サマリー拡張、`0021_phase7b_calibration_compatibility`で厳密校正条件を追加する。内容fingerprintと校正状態fingerprintを分離し、関連校正だけをstale化する。Optuna等の自動探索、複数jobの自動起動、lossだけによる品質判定は対象外とする

## Phase 8A実装状況: Danbooru検索・取得候補管理・取得計画

- `ImageSourceAdapter` Protocol、Danbooruアダプター、Fake adapter、固定source registryを追加した。queryはタグ・rating・数値・拡張子を正規化し、固定query fingerprintを生成する。任意URL、任意rating文字列、制御文字、bool数値は受け付けない。
- `ExternalImagePost`、`ImageSourceSearch`、検索結果、`ImageAcquisitionPlan`、plan item、外部postと既存ImageAssetのlinkをAlembic `0027_phase8a_image_acquisition`へ追加した。source type + external post IDとplan fingerprintにunique制約を持たせ、MD5やURL単独では同一postと判定しない。
- Danbooru APIは固定HTTPS host、JSON content type、応答サイズ上限、redirect拒否、環境変数認証、単一workerのmonotonic rate limiter、429／408／5xx／timeoutのbounded retry、秒数およびHTTP-date形式のRetry-After、待機中キャンセルを使用する。raw responseと認証情報は保存しない。
- 検索はqueued/running/completed/partially_failed/failed/canceled/staleを保存し、`0028_phase8a_claim_reservations`でworker generationとclaim tokenを追加した。`0029_phase8a_page_checkpoints`でrequest cursor、完了理由、worker世代をまたぐcursor checkpoint履歴を追加し、claim、heartbeat、cursor、候補保存、終端更新はsearch ID・worker ID・claim tokenの条件付き更新で保護する。
- ページ処理ではAPI request countとcommitted page countを分離する。APIへ渡すrequest cursorを先に保存し、ページ内候補のupsert、accepted／excluded／returned counter、committed next cursor、page count、cursor履歴を同一transactionで確定する。途中停止やclaim喪失時はtransactionをrollbackし、stale searchは最後の完全checkpointまたは処理中ページのrequest cursorから冪等に再実行する。候補数上限到達時は`candidate_limit`としてcompletedにし、未処理残りのnext cursorは保存しない。候補は`MISSING_FILE_URL`等の固定除外理由と利用可否を保存する。
- 候補のexternal postとsearch resultはDBの一意制約付きupsertで冪等に保存する。取得計画にはsource type + external post IDのreservationを作成し、別planとの同時確定をDBレベルで防止する。
- 取得計画のpreview fingerprintには検索fingerprint、adapter version、plan状態、選択件数、候補メタデータを含める。confirm時にDBを再検証し、確定済みplanは不変・冪等で、画像ダウンロードは行わない。UIは外部画像を表示せず、ratingと除外理由を日本語で表示する。
- Phase 8Bの画像取得、ファイル検証、ImageRecord登録、サムネイル、ZIP、他source、tagger／dataset／Drive連携は未実装である。
