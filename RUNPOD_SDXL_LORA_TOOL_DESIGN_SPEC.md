# SDXL LoRA Dataset & Training Studio for RunPod
## 設計仕様書

- 文書版数: 1.0
- 作成日: 2026-07-19
- 対象環境: RunPod GPU Pod
- 対象モデル: SDXL系チェックポイント
- 想定用途: アニメ・ゲームキャラクター、画風、衣装、物体などのLoRA作成支援
- 仮称: **LoRA Dataset & Training Studio（LDTS）**

---

## 1. 目的

本ツールは、RunPod上でSDXL系LoRAを作成する際に必要となる以下の作業を、1つのWeb UIから一貫して行えるようにする。

1. 学習画像の収集
2. 重複・低品質画像の検出
3. 学習画像の目視選別
4. 自動タグ付けとタグ編集
5. データセット構築
6. 学習元モデルと学習条件の選択
7. 推奨パラメータの自動算出
8. LoRA学習の実行・監視・再開
9. 学習途中および完成後の比較評価
10. データセット・設定・成果物の保存と再利用

初期対応の画像取得元はDanbooruとするが、将来的に他の画像取得元、ローカルアップロード、Google Drive、S3互換ストレージなどを追加できる構造とする。

---

## 2. 基本方針

### 2.1 ユーザー体験

専門的な学習パラメータを理解していない利用者でも、以下の手順でLoRAを作成できることを目標とする。

> プロジェクト作成 → 画像検索・取得 → 自動検査 → 画像選択 → タグ調整 → 学習設定 → 学習 → 比較評価 → 出力

一方で、上級者向けにkohya-ss / sd-scriptsの主要パラメータを手動設定できる「詳細設定モード」も提供する。

### 2.2 非破壊処理

取得した原本画像、加工後画像、キャプション、選択状態、除外理由を分離して保存する。

自動処理によって画像を物理削除せず、原則として状態を以下のいずれかに変更する。

- 採用候補
- 採用
- 保留
- 自動除外
- 手動除外
- 重複代表
- 重複画像
- 処理エラー

ユーザーがいつでも除外を取り消せること。

### 2.3 再現性

以下をプロジェクト単位で保存する。

- 取得元と検索条件
- 取得日時
- 元ページURLまたは投稿ID
- 原画像ハッシュ
- 画像処理設定
- タガーモデルと閾値
- キャプション編集履歴
- 学習元モデルの識別情報
- 学習設定
- 乱数シード
- 実行コマンド
- 使用ライブラリのバージョン
- GPU情報
- 学習ログ
- サンプル画像
- 出力LoRAのハッシュ

同じプロジェクト設定から、可能な範囲で同じ学習を再実行できること。

---

## 3. 対象範囲

### 3.1 初期リリースで対応する範囲

- Danbooruからの画像検索・取得
- ローカルPCからの画像アップロード
- 画像重複・類似画像の検出
- 基本的な品質判定
- 画像一覧・比較・選択
- WD系タガーによる自動タグ付け
- タグの一括削除・置換・追加
- トリガーワード付与
- SDXL系LoRA学習
- kohya-ss / sd-scriptsを利用した学習実行
- 推奨パラメータ生成
- 手動パラメータ編集
- 学習進捗表示
- 定期サンプル生成
- 成果物ダウンロード
- プロジェクトのエクスポート・インポート

### 3.2 将来拡張

- Gelbooru、Safebooru、e621等の追加コネクタ
- Pixiv等、利用規約やAPI条件を満たす取得元への対応
- Google Drive、S3、Cloudflare R2等との同期
- Civitai / Hugging Faceからのモデル取得
- OneTrainer等、別学習バックエンド
- Flux、SD 1.5、Illustrious、NoobAI等のモデル系列別プリセット
- セグメンテーションや背景除去
- 顔・衣装・構図のバランス最適化
- 複数概念LoRA
- LoRAマージ、差分比較
- 自動ハイパーパラメータ探索
- 複数GPU・ジョブキュー
- RunPod APIによるPod作成・停止の自動化

---

## 4. システム構成

## 4.1 推奨アーキテクチャ

本ツールは自分専用として利用するため、初期版はGradioをRunPod上で起動し、ローカルPCのブラウザからRunPod HTTP Proxy経由でアクセスする構成とする。処理本体はすべてRunPod上で動作し、ローカルPCは操作画面としてのみ利用する。

```text
[ローカルPCのブラウザ]
      |
      | HTTPS / RunPod HTTP Proxy
      v
[RunPod上のGradio Web UI :7860]
      |
      +-------------+--------------+----------------+
      |             |              |                |
      v             v              v                v
[Downloader]   [Image QA]      [Tagger]       [Training Runner]
 Danbooru      hash/CLIP       WD Tagger       sd-scripts
 adapters      quality         caption rules   subprocess
      |
      v
[RunPod一時作業領域]
 /workspace/ldts-runtime
      | 学習正常完了時のみ rclone copy
      v
[Google Drive]
 models / projects / datasets / outputs
```

コードはGitで管理し、RunPod起動時にGitHub等のプライベートリポジトリから`git clone`または`git pull`する。Google Driveはコード配布には使わず、モデル、画像、キャプション、設定、完成LoRA、ログ等の保管先として利用する。

### 4.2 推奨技術

| 区分 | 推奨 |
|---|---|
| Web UI | Gradio Blocks |
| 実装言語 | Python 3.11系 |
| コード管理 | Git + GitHub等のプライベートリポジトリ |
| 初期DB | SQLite |
| 長時間処理 | `subprocess.Popen` + 独自ジョブ管理 |
| 将来の非同期処理 | 必要になった場合のみRQ / Celery |
| 画像処理 | Pillow、OpenCV、imagehash |
| 類似度判定 | pHash + CLIP埋め込み |
| タグ付け | WD 1.4系Taggerを交換可能な形で実装 |
| 学習 | kohya-ss / sd-scriptsの`sdxl_train_network.py` |
| Google Drive同期 | rclone |
| 設定形式 | TOML + JSON |
| ログ | structlogまたは標準logging |
| コンテナ | 初期はRunPod PyTorchテンプレート、安定後にDocker化 |
| テスト | pytest、Gradio UIの主要動作テスト |
| 型・品質 | mypy、ruff |

### 4.3 長時間処理の分離

学習処理をGradioのイベント関数内で同期実行しない。`sdxl_train_network.py`は別プロセスとして起動し、PID、標準出力、終了コード、進捗、停止要求をジョブ管理層で扱う。

ブラウザやGradio画面を閉じても、RunPod Podが稼働している限り学習を継続できること。再接続後に状態とログを復元できること。

---

## 5. プロジェクト構造

```text
/workspace/ldts/
├── app/
├── database/
│   └── ldts.sqlite3
├── projects/
│   └── {project_id}/
│       ├── project.json
│       ├── sources/
│       │   └── source_manifest.jsonl
│       ├── originals/
│       ├── previews/
│       ├── processed/
│       ├── captions/
│       ├── dataset/
│       ├── configs/
│       │   ├── dataset.toml
│       │   └── training.toml
│       ├── models/
│       ├── outputs/
│       ├── samples/
│       ├── logs/
│       └── reports/
├── shared_models/
│   ├── checkpoints/
│   ├── vae/
│   ├── taggers/
│   └── clip/
└── cache/
```

原画像は`originals`に保持し、リサイズ・クロップ等を行った画像は`processed`または学習時キャッシュに保存する。

---

## 6. 主要データモデル

### 6.1 Project

| 項目 | 内容 |
|---|---|
| id | UUID |
| name | プロジェクト名 |
| concept_type | character / style / costume / object / other |
| trigger_words | トリガーワード配列 |
| model_family | sdxl / illustrious_sdxl等 |
| status | draft / preparing / ready / training / completed / failed |
| created_at | 作成日時 |
| updated_at | 更新日時 |
| settings_version | 設定スキーマ版 |

### 6.2 SourceDefinition

| 項目 | 内容 |
|---|---|
| source_type | danbooru / upload / future connector |
| query | 検索条件 |
| credentials_ref | APIキー等への安全な参照 |
| rating_policy | safe / sensitive / questionable / explicitの扱い |
| max_items | 最大取得件数 |
| sort_order | score / date / random等 |
| filters | サイズ、形式、スコア等 |

### 6.3 ImageAsset

| 項目 | 内容 |
|---|---|
| id | UUID |
| source_type | 取得元種別 |
| source_post_id | 投稿ID |
| source_url | 投稿ページ |
| original_url | 原画像URL |
| local_path | 保存先 |
| sha256 | 完全重複判定 |
| phash | 視覚的近似判定 |
| width / height | 寸法 |
| file_size | ファイルサイズ |
| mime_type | 形式 |
| source_tags | 取得元タグ |
| detected_tags | 自動タグ |
| final_tags | 学習用タグ |
| quality_scores | 各品質指標 |
| duplicate_group_id | 類似グループ |
| selection_state | 採用状態 |
| exclusion_reasons | 除外理由配列 |
| manual_override | 自動判定の上書き有無 |

### 6.4 TrainingRun

| 項目 | 内容 |
|---|---|
| id | UUID |
| project_id | 対象プロジェクト |
| config_snapshot | 実行時設定 |
| dataset_snapshot_id | データセット版 |
| base_model | 学習元モデル |
| status | queued / running / paused / completed / failed / canceled |
| command | 実行コマンド |
| pid | プロセスID |
| current_epoch | 現在エポック |
| current_step | 現在ステップ |
| loss | 最新loss |
| output_paths | 出力一覧 |
| started_at / ended_at | 開始・終了 |
| error_summary | エラー概要 |

---

## 7. 画像取得機能

## 7.1 取得元アダプター

画像取得元は共通インターフェースを実装する。

```python
class ImageSourceAdapter(Protocol):
    source_name: str

    async def validate_credentials(self) -> ValidationResult: ...

    async def search(self, request: SearchRequest) -> AsyncIterator[SourceItem]: ...

    async def fetch_metadata(self, source_id: str) -> SourceItem: ...

    async def download(self, item: SourceItem, destination: Path) -> DownloadResult: ...
```

Danbooru固有の処理をサービス本体に直書きせず、`DanbooruAdapter`として分離する。

## 7.2 Danbooru検索条件

UIから以下を指定できること。

- タグ検索
- 除外タグ
- rating
- 最低スコア
- 最低解像度
- 最大件数
- 投稿期間
- 並び順
- 静止画のみ
- 対象拡張子
- 透かし・テキスト・漫画ページ等の除外候補
- AI生成タグの許可・除外
- 単独キャラクター優先
- 複数人物画像の許可・除外
- API認証情報

### 7.3 取得時の必須処理

- robots.txt、API制限、利用規約を尊重する
- リクエスト間隔と同時接続数を制限する
- HTTP 429時は指数バックオフする
- 途中停止と再開に対応する
- 既取得投稿を再ダウンロードしない
- URLだけでなく投稿IDとメタデータを保存する
- 削除済み・取得失敗を記録する
- 不正な拡張子や破損ファイルを検出する
- 一時ファイルに保存後、検証できた場合のみ確定名へ変更する

### 7.4 権利・利用条件表示

画像取得画面に以下を表示する。

- 取得元の利用規約に従う必要があること
- 学習・公開・販売の可否は、画像や作品ごとの権利条件に依存すること
- 取得元タグや投稿情報が、利用許諾を保証するものではないこと
- ユーザー自身が利用権限を確認する必要があること

---

## 8. 重複・類似画像検出

## 8.1 多段階判定

### 第1段階: 完全重複

- SHA-256
- ファイルバイト列が完全一致した画像を同一とする

### 第2段階: 近似重複

- pHash、dHash等
- リサイズ、軽微な色差、圧縮差を検出する
- ハミング距離により閾値を設定する

### 第3段階: 意味的類似

- CLIP等の画像埋め込み
- クロップ違い、文字入れ違い、軽微な加工違いを候補としてグループ化する
- 自動削除ではなく、類似候補としてUIに表示する

## 8.2 代表画像選定

類似グループ内では以下を加点し、代表候補を決定する。

- 解像度が高い
- 圧縮ノイズが少ない
- 透かしが少ない
- 対象キャラクターが大きく写っている
- 顔が明瞭
- 検索対象タグとの一致度が高い
- 投稿スコアが高い

最終選択はユーザーが変更できる。

---

## 9. 画像品質評価

## 9.1 自動評価項目

- 最小幅・最小高さ
- 極端な縦横比
- 画像破損
- 強いぼけ
- 極端な暗さ・白飛び
- JPEGブロックノイズ
- 透かし
- 大量の文字
- 漫画のコマ割り
- スクリーンショットUI
- 対象が小さすぎる
- 顔の欠損・極端な見切れ
- 人数
- 単色画像
- 低情報量
- AI生成疑いタグ
- NSFW区分
- 指定キャラクターとの意味的一致度
- 異なる衣装・別キャラクター混入の疑い

### 9.2 判定方法

品質判定は、単一の総合点だけで除外しない。

```text
quality_score
├── technical_score
├── subject_visibility_score
├── aesthetic_score
├── text_watermark_score
├── concept_relevance_score
└── safety_policy_score
```

各指標と除外理由をユーザーに表示する。

### 9.3 初期の自動除外例

- 画像破損
- 幅または高さが指定値未満
- 完全重複
- 明らかな単色画像
- 設定上禁止されたrating
- ユーザーが指定した必須タグを満たさない

ぼけ、構図、透かし、類似画像など、誤判定の可能性がある項目は原則「除外候補」とする。

---

## 10. 画像選別UI

## 10.1 一覧表示

- サムネイルグリッド
- 画像サイズ
- 取得元
- 投稿スコア
- rating
- 採用状態
- 品質スコア
- 除外理由
- 類似グループ
- タグ数
- キャプション警告
- 複数選択チェック

## 10.2 フィルター

- 採用状態
- 除外理由
- 取得元
- rating
- 解像度
- 縦横比
- 類似グループ
- 品質スコア範囲
- 特定タグの有無
- 人数
- 手動確認済み／未確認
- タグ付け済み／未処理

## 10.3 詳細ビュー

画像クリック時に以下を表示する。

- 原寸プレビュー
- 取得元情報
- 自動品質判定の内訳
- 類似画像横並び比較
- 取得元タグ
- 自動タグ
- 最終キャプション
- 採用・保留・除外操作
- 画像回転
- 任意クロップ範囲
- 簡易マスク指定
- メモ

## 10.4 一括操作

- 採用
- 除外
- 保留
- タグ付け
- タグ追加
- タグ削除
- タグ置換
- トリガーワード変更
- 再品質判定
- 削除候補を復元
- データセットから外す

---

## 11. 自動タグ付け

## 11.1 タガー抽象化

```python
class TaggerAdapter(Protocol):
    model_name: str

    def load(self) -> None: ...

    def predict(self, image: Image.Image, settings: TaggerSettings) -> TagResult: ...

    def unload(self) -> None: ...
```

初期実装はWD 1.4系Taggerを想定する。

## 11.2 タグ設定

- general tag閾値
- character tag閾値
- rating tagの保持・削除
- アンダースコアを空白へ変換
- エスケープ処理
- タグ順序
- 最大タグ数
- blacklist
- whitelist
- 先頭固定タグ
- 末尾固定タグ
- キャラクター名タグの扱い
- 取得元タグとのマージ方式

## 11.3 タグ編集フロー

タグ処理は以下の順序で行う。LoRA作成前に、採用画像全体で使用されているタグを集計し、ユーザーが残すタグと削除するタグを手動で確定する。

```text
1. タガー出力を取得
2. 取得元タグを必要に応じてマージ
3. 閾値未満のタグを除外候補にする
4. タグを正規化し、同義語を統合する
5. 採用画像内のタグ出現回数を集計する
6. 出現回数の多い順にタグ選択画面へ表示する
7. ユーザーがチェックボックスで保持・削除を指定する
8. 確定した削除対象タグを全採用画像から削除する
9. トリガーワードを付与する
10. タグ順序を整列する
11. 最終キャプションを保存する
```

### 11.4 学習前タグ選択画面

LoRA学習を開始する前に、採用画像に含まれるタグの集計結果を一覧表示する。初期表示は出現画像数の多い順とし、各タグについてチェックボックスで学習用キャプションに残すか削除するかを手動で選択できること。

#### 表示項目

- 保持チェックボックス
- タグ名
- 出現画像数
- 採用画像全体に対する出現率
- タグカテゴリ
- タガー信頼度の平均値
- そのタグを含む画像のサムネイル例
- 自動タグ、取得元タグ、手動追加タグの区分

#### 初期状態

- チェックあり: 最終キャプションに残す
- チェックなし: 全採用画像の最終キャプションから削除する
- 原則として全タグをチェックありで表示し、ユーザーが不要タグのチェックを外す
- blacklistに登録済みのタグはチェックなしで表示するが、ユーザーが再度チェックを入れられる
- トリガーワードおよび先頭固定タグは固定表示とし、誤操作防止のため通常のタグとは区別する

#### 並び替えと絞り込み

- 出現画像数の多い順・少ない順
- タグ名順
- 出現率順
- タグカテゴリ別
- チェックあり・チェックなし
- キーワード検索
- 指定回数以上または以下のタグ
- 自動タグ、取得元タグ、手動追加タグ

#### 操作

- すべて残す
- すべて削除する
- 表示中のタグだけ残す
- 表示中のタグだけ削除する
- 選択状態をプリセットとして保存する
- 前回の選択状態を読み込む
- 選択確定前に、削除後のキャプションを画像単位でプレビューする
- 確定後も学習開始前であれば選択を変更して再適用できる

#### 確定時の処理

タグ選択を確定すると、チェックが外れているタグを全採用画像の最終キャプションから削除する。元の自動タグ結果は保持し、最終キャプションのみを更新する非破壊処理とする。タグ選択状態、集計対象となったデータセット版、確定日時を保存する。

### 11.5 トリガーワード

- プロジェクトに複数登録可能
- 画像種別ごとに異なるトリガーを付与可能
- 常に先頭へ付与
- 指定確率で付与
- 衣装別トリガー
- 表情・髪型等の補助トリガー
- 予約語・既存タグとの衝突警告
- 使用文字種の検証
- 学習元モデルのタグ体系に応じた警告

---

## 12. データセット分析

学習開始前にデータセット診断レポートを生成する。

### 12.1 診断項目

- 採用画像枚数
- 解像度分布
- 縦横比分布
- 重複率
- タグ出現頻度
- 画像あたりタグ数
- トリガーワード付与率
- 単独／複数人物比率
- 顔アップ／上半身／全身比率
- 正面／横顔／背面の偏り
- 衣装比率
- 背景の偏り
- rating分布
- 取得元分布
- 推定総ステップ
- 過学習リスク
- データ不足警告
- タグ表記揺れ
- キャプション空欄
- 読み込み不能画像

### 12.2 改善提案

例:

- 顔アップが多すぎるため全身画像を追加する
- 特定衣装が80%を占めており、衣装がキャラクター特徴として固定される可能性がある
- 背景タグを残しすぎている
- トリガーワードと既存キャラクタータグが併存している
- 類似画像が多く、実質的なデータ多様性が低い
- 画像枚数に対して総ステップが多い
- 学習元モデルと画像の画風差が大きい可能性がある

---

## 13. 学習元モデル管理

## 13.1 モデル指定方法

- RunPod内のファイル選択
- Hugging Face形式ディレクトリ
- `.safetensors`ファイル
- URLから取得
- 将来: Civitai / Hugging Face連携

### 13.2 モデル検証

- ファイル存在
- SHA-256
- SDXL互換性
- モデル形式
- VAE有無
- prediction type
- clip skip相当設定の適合性
- ライセンス・メモ
- 保存容量
- 既存キャッシュとの重複

### 13.3 モデルレジストリ

```text
表示名
ファイルパス
モデル系列
ベースモデル
ハッシュ
推奨解像度
推奨タグ形式
VAE
prediction type
備考
```

Illustrious系など、SDXL派生モデルごとのプリセットは、学習バックエンド設定と分離して管理する。

---

## 14. 推奨学習パラメータ生成

## 14.1 入力情報

- 概念種別
- 画像枚数
- 実質画像枚数
- 類似度
- 解像度分布
- 学習元モデル系列
- GPU VRAM
- 目標LoRAサイズ
- 品質優先／速度優先
- text encoder学習の有無
- repeats
- epoch
- batch size
- network dimension
- optimizer
- caption dropout
- augmentation設定

## 14.2 推奨ロジックの方針

固定値を提示するだけでなく、以下を計算する。

```text
steps_per_epoch =
  ceil(image_count × repeats / batch_size / gradient_accumulation_steps)

total_steps =
  steps_per_epoch × epochs
```

実質画像枚数は、類似画像の多さを考慮して補正する。

```text
effective_image_count =
  selected_image_count × diversity_factor
```

### 14.3 初期プリセット例

数値は絶対値ではなく初期候補であり、モデル系列やデータ内容に応じて変更可能とする。

#### キャラクターLoRA・少量

| 項目 | 初期候補 |
|---|---|
| 画像枚数 | 15～40枚 |
| 解像度 | 1024 |
| bucket | 有効 |
| network_dim | 16～32 |
| network_alpha | dimの半分～同値 |
| optimizer | AdamW8bit |
| UNet LR | 1e-4前後から候補生成 |
| text encoder | 初期設定では無効 |
| batch size | VRAMに応じて1～4 |
| 総ステップ | 約1,000～3,000を起点に診断 |
| save | epochごと |
| sample | epochごと |

#### キャラクターLoRA・中量

| 項目 | 初期候補 |
|---|---|
| 画像枚数 | 40～150枚 |
| network_dim | 32～64 |
| 総ステップ | 約2,000～6,000を起点 |
| caption dropout | データに応じて検討 |
| tag dropout | 必要に応じて低率 |
| noise offset | モデル系列プリセットに従う |

#### 画風LoRA

| 項目 | 初期候補 |
|---|---|
| 画像枚数 | 50枚以上を推奨 |
| network_dim | 32～128 |
| キャプション | 内容タグを比較的詳しく保持 |
| トリガー | 画風用トリガー |
| text encoder | 原則慎重に扱う |
| 評価 | 複数の被写体・構図で評価 |

### 14.4 VRAM別調整

- batch size
- gradient checkpointing
- mixed precision
- cache latents
- cache text encoder outputs
- optimizer
- gradient accumulation
- persistent data loader workers
- xformers / SDPA
- lowram
- VAE batch size

推奨値算出時に、推定VRAM使用量と安全マージンを表示する。

### 14.5 推奨根拠の表示

各自動設定に理由を表示する。

例:

```text
network_dim = 32
理由: キャラクター学習、採用画像38枚、特徴の複雑度は中程度。
64以上では学習容量に対してデータ量が少なく、過学習リスクが上がるため。
```

### 14.6 危険設定の警告

- 総ステップが多すぎる
- text encoder LRが高すぎる
- network_dimが画像枚数に対して大きすぎる
- batch sizeとVRAMの不整合
- 保存先容量不足
- トリガーワードが全画像に入っていない
- キャプションが空
- 学習元モデルが未確認
- 学習元と推論予定モデルの系列が異なる
- 全画像がほぼ同じ構図
- 同じ画像を過剰なrepeatsで学習
- 明示的コンテンツが混在している

---

## 15. 手動学習設定

### 15.1 基本設定

- output name
- pretrained model
- VAE
- train data directory / dataset config
- resolution
- enable bucket
- min/max bucket resolution
- batch size
- epochs
- max train steps
- repeats
- save frequency
- seed
- mixed precision
- save precision

### 15.2 LoRA設定

- network module
- network dim
- network alpha
- conv dim
- conv alpha
- dropout
- train UNet only
- text encoder training
- block weights
- LoRA+等の対応可能な追加方式

### 15.3 Optimizer / Scheduler

- optimizer type
- optimizer arguments
- UNet learning rate
- text encoder learning rate
- LR scheduler
- warmup steps / ratio
- gradient accumulation
- max grad norm

### 15.4 メモリ・性能

- gradient checkpointing
- cache latents
- cache latents to disk
- cache text encoder outputs
- persistent data loader workers
- max data loader workers
- xformers / SDPA
- full fp16 / bf16
- lowram
- VAE batch size

### 15.5 データ拡張

- flip augmentation
- color augmentation
- random crop
- caption dropout
- tag dropout
- shuffle caption
- keep tokens
- token warmup
- face crop augmentation
- min SNR gamma
- noise offset
- multires noise
- IP noise gamma

モデル系列に適さない設定は非表示または警告付きにする。

---

## 16. 学習実行

## 16.1 実行前チェック

- データセットが固定版として保存されている
- 全採用画像を読み込める
- 全キャプションが存在する
- トリガーワード条件を満たす
- 学習元モデルを読み込める
- 出力先に書き込める
- ディスク容量が十分
- GPUを認識している
- CUDA / PyTorch互換性
- 必要ライブラリが存在
- 同一GPUで競合ジョブが動いていない
- 推定所要VRAMが上限を超えていない
- 実行コマンドを生成できる

## 16.2 実行方式

設定からTOMLを生成し、`sdxl_train_network.py`をサブプロセスとして起動する。

- 標準出力・標準エラーをリアルタイム取得
- ログをファイル保存
- 正常終了コードを確認
- PID管理
- 停止要求はSIGINT → 一定時間後SIGTERM
- 異常終了時は直近ログと原因候補を表示
- 実行コマンドを保存
- shell文字列連結ではなく引数配列で実行
- 任意コマンド注入を防止

## 16.3 進捗表示

- 現在のepoch
- 現在step / 総step
- loss
- learning rate
- 経過時間
- GPU使用率
- VRAM使用量
- 温度
- 保存済みチェックポイント
- 最新サンプル
- ログ末尾
- 推定残り時間は参考表示

## 16.4 中断・再開

- 学習中断
- 最後のstateから再開
- 任意epochのstateから再開
- 設定変更後は別Runとして複製
- 再開不能な設定変更を警告
- Pod停止前にstate保存を促す

---

## 17. サンプル生成と評価

## 17.1 学習中サンプル

- epochごとまたは指定stepごと
- 固定seed
- 固定プロンプト
- 複数LoRA強度
- ネガティブプロンプト
- 解像度
- sampler / scheduler
- 保存先

## 17.2 評価プロンプトセット

プロジェクトごとに評価テンプレートを保存する。

キャラクター例:

- 顔アップ
- 上半身
- 全身
- 正面
- 横顔
- 異なる表情
- 異なる衣装
- 単色背景
- 複雑な背景
- 他キャラクターとの組み合わせ
- トリガーワードなし

画風例:

- 人物
- 動物
- 風景
- 室内
- 昼
- 夜
- 単純構図
- 複雑構図

## 17.3 比較UI

- epoch別サンプルを同一seedで横並び
- base modelとの比較
- LoRA強度別比較
- 画像スライダー
- お気に入り
- 評価メモ
- 採用epoch指定
- 過学習兆候のチェック項目

### 17.4 自動評価補助

完全自動の良否判定には依存しないが、補助指標として以下を利用可能とする。

- トリガーワード使用時の対象一致度
- トリガーワードなしでの漏出度
- 学習画像との過度な近似
- 顔の一貫性
- 画像多様性
- CLIP類似度
- aesthetic score

---

## 18. 成果物

### 18.1 出力

- LoRA `.safetensors`
- 学習設定TOML
- dataset TOML
- キャプション一式
- 学習ログ
- サンプル画像
- データセット診断レポート
- 学習結果レポート
- モデルカードMarkdown
- SHA-256一覧
- プロジェクトエクスポートZIP

### 18.2 モデルカード

以下を自動生成する。

- LoRA名
- 対象モデル系列
- 推奨ベースモデル
- トリガーワード
- 推奨LoRA強度
- 使用画像枚数
- 学習設定の概要
- 推奨プロンプト
- 注意事項
- 学習日
- 使用ツール版
- 権利・公開範囲に関するユーザー記入欄

---

## 19. 画面構成

### 19.1 ダッシュボード

- プロジェクト一覧
- 状態
- 採用画像枚数
- 最終学習
- 最新成果物
- ストレージ使用量
- GPU情報
- 実行中ジョブ

### 19.2 プロジェクト画面

左ナビゲーション:

1. 概要
2. 画像取得
3. 品質・重複検査
4. 画像選別
5. タグ編集
6. データセット分析
7. モデル
8. 学習設定
9. 学習実行
10. 評価
11. 成果物
12. 履歴
13. プロジェクト設定

### 19.3 システム設定

- 保存先
- キャッシュ上限
- Danbooru認証
- Hugging Faceトークン
- タガーモデル
- 学習バックエンド
- 同時ジョブ数
- ログレベル
- UIテーマ
- ネットワーク公開設定
- バージョン情報

---

## 20. API案

```text
POST   /api/projects
GET    /api/projects
GET    /api/projects/{id}
PATCH  /api/projects/{id}

POST   /api/projects/{id}/sources
POST   /api/projects/{id}/acquisition-jobs
GET    /api/jobs/{job_id}
POST   /api/jobs/{job_id}/cancel

GET    /api/projects/{id}/images
GET    /api/images/{image_id}
PATCH  /api/images/{image_id}
POST   /api/images/bulk-update

POST   /api/projects/{id}/duplicate-scan
POST   /api/projects/{id}/quality-scan
POST   /api/projects/{id}/tagging-jobs
POST   /api/projects/{id}/caption-rules/apply

GET    /api/projects/{id}/dataset-report
POST   /api/projects/{id}/dataset-snapshots

GET    /api/models
POST   /api/models/register
POST   /api/models/download
POST   /api/models/validate

POST   /api/projects/{id}/training/recommend
POST   /api/projects/{id}/training/validate
POST   /api/projects/{id}/training/runs
GET    /api/training/runs/{run_id}
POST   /api/training/runs/{run_id}/stop
POST   /api/training/runs/{run_id}/resume

GET    /api/training/runs/{run_id}/samples
GET    /api/projects/{id}/artifacts
POST   /api/projects/{id}/export
```

WebSocketまたはServer-Sent Eventsでジョブ進捗を配信する。

---

## 21. セキュリティ

- 初期状態では認証なしの公開ポートにしない
- RunPod Proxyまたは認証付きリバースプロキシを使用
- APIキー・トークンをDBへ平文保存しない
- `.env`または秘密管理機構を利用
- UI上では秘密値をマスク
- URL取得時のSSRF対策
- 保存パスのディレクトリトラバーサル対策
- アップロード拡張子だけでなく実体を検査
- 画像デコーダのリソース制限
- ZIP展開時のZip Slip対策
- 任意シェルコマンド実行を許可しない
- 学習引数は許可リスト方式
- CSRF、XSS、CORS設定
- 操作監査ログ
- NSFW画像プレビューのぼかし設定
- 未成年を示す可能性のある明示的コンテンツは取得・学習対象にしないための警告・遮断設計
- 利用者が法令・サービス規約・著作権・肖像権を確認する前提を明示

---

## 22. RunPod対応

### 22.1 保存方針

コードとデータの保存先を明確に分離する。

| 対象 | 正式な保存先 | RunPod上での扱い |
|---|---|---|
| ツールのソースコード | Gitリポジトリ | 起動時にcloneまたはpull |
| 学習元モデル | Google Drive | 学習開始前にRunPodへコピー |
| 採用した学習画像 | Google Drive | 編集・学習中はRunPod一時領域に保持 |
| タグ付け後テキスト | Google Drive | 編集・学習中はRunPod一時領域に保持 |
| データセット設定 | Google Drive | 実行中はRunPod一時領域に保持 |
| 学習途中のstate・checkpoint | RunPodのみ | 学習完了までは同期しない |
| 完成LoRA | Google Drive | 正常完了後に同期 |
| 完了時ログ・サンプル・レポート | Google Drive | 正常完了後に同期 |

RunPod内の一時作業領域は以下を標準とする。

```text
/workspace/ldts-runtime
```

Google Drive側の標準構成は以下とする。

```text
gdrive:SDXL-LoRA-Studio/
├── models/
├── projects/
│   └── {project_name}/
│       ├── originals/
│       ├── selected_images/
│       ├── captions/
│       ├── configs/
│       ├── manifests/
│       ├── samples/
│       ├── logs/
│       └── outputs/
└── archives/
```

学習途中のキャッシュ、latent、state、epoch別LoRA等はRunPod内だけに保存し、学習正常完了後に必要な完成物だけをGoogle Driveへ`rclone copy`する。

### 22.2 コード配布と起動

コードはGitで管理する。RunPod起動時にプライベートリポジトリから取得する。

```bash
APP_DIR=/workspace/ldts-app

if [ ! -d "$APP_DIR/.git" ]; then
    git clone "$LDTS_GIT_REPOSITORY" "$APP_DIR"
else
    git -C "$APP_DIR" pull --ff-only
fi

cd "$APP_DIR"
exec python app.py
```

GradioはRunPod上のポート7860で起動する。

```python
app.launch(
    server_name="0.0.0.0",
    server_port=7860,
    share=False,
)
```

RunPodテンプレートではHTTPポート7860のみを公開する。`share=True`によるGradio共有URLは使用しない。

### 22.3 GPU自動検出

起動時に以下を取得する。

- GPU名
- GPU数
- VRAM
- CUDA version
- PyTorch CUDA version
- bf16対応
- xformers / SDPA可否

Torchのdevice indexはプロセスから見えるlogical indexとして扱い、
CUDA_VISIBLE_DEVICESの数値tokenをphysical indexとして直接比較しない。
physical index、GPU UUID、architecture、compute capability、total VRAMは固定queryの
nvidia-smi inventoryとTorch device UUIDを照合して確定する。UUID prefixは一意に解決
できる場合だけ受け付け、曖昧・不正な指定はunverifiedとして扱う。
torch.cuda.mem_get_info()の戻り値は(free, total)であり、free VRAMは変動値、total VRAMは
GPU identityに紐づく固定値として保存する。

開始前に複数GPUから実行GPUを一意に選べない場合、推奨付きジョブは開始しない。
手動ジョブでは対象PIDのcompute-process queryが単一UUIDを返した場合に限り、
TrainingJobSelectedGpuとしてruntime identityを別途保存する。summaryとcalibrationの
GPU属性は確定したruntime identityに対応する同一physical inventoryから取得し、
pre-startのvisible GPU集合から別GPUの属性をfallbackしない。
selected GPU identityは最初の確定後immutableとし、後続のGPU変更は初回identityを
上書きせず、最終観測UUID・変更時刻・変更回数・warning codeを限定的に監査保存する。
GPU変更またはselected GPUとmemory aggregateのUUID不一致がある実績は、速度・VRAM校正へ
利用せず、関連するcalibrationをstale化する。

取得結果を推奨パラメータ生成へ渡す。

### 22.4 ポート

- Web UIのみ外部公開
- Redis、DB、内部APIを不用意に公開しない
- Web UIに認証を付ける
- 外部ポート番号変更を前提に、画面内URLを固定しない

### 22.5 ストレージ節約

- VAE latent cacheを削除可能
- タガーモデル共有
- チェックポイントをプロジェクトごとに複製しない
- ハードリンクまたは参照方式
- 古いstateの保持数を設定
- サンプル生成枚数制限
- キャッシュ容量上限
- 削除前に対象容量を表示

### 22.6 Google Drive同期

Google Drive連携にはrcloneを使用する。rcloneはコード配布ではなく、モデル、データセット、成果物の転送に利用する。

#### 学習開始前

- 指定された学習元モデルをGoogle DriveからRunPodへコピーする
- 既存プロジェクトを再開する場合は、画像、キャプション、設定をRunPodへコピーする
- 同名・同一ハッシュのモデルがRunPodに存在する場合は再取得しない
- 転送完了後にサイズまたはハッシュを検証する

例:

```bash
rclone copy \
  "gdrive:SDXL-LoRA-Studio/models/${MODEL_NAME}" \
  "/workspace/ldts-runtime/models/${MODEL_NAME}" \
  --checksum \
  --retries 5
```

#### 学習中

- Google Driveへの継続同期は行わない
- epochごとのLoRA、state、latent cache、ログ追記はRunPod内にのみ保存する
- Google Drive API制限、転送遅延、同期競合によって学習処理を妨げない

#### 学習完了時

以下をGoogle Driveへ同期する。

- 完成LoRA `.safetensors`
- 学習に採用した画像
- タグ付け・手動編集後のテキストファイル
- dataset TOML
- training TOML
- 取得元manifest
- 学習ログ
- サンプル画像
- 診断・評価レポート
- SHA-256一覧
- 完了manifest

同期には原則として`rclone copy`を使用する。`rclone sync`はGoogle Drive側だけにあるファイルを削除する可能性があるため、通常処理では使用しない。

```bash
rclone copy \
  "/workspace/ldts-runtime/projects/${PROJECT_NAME}/export" \
  "gdrive:SDXL-LoRA-Studio/projects/${PROJECT_NAME}" \
  --checksum \
  --retries 5 \
  --create-empty-src-dirs
```

### 22.7 完了同期とPod自動終了

Gradioの学習開始画面に以下を設ける。

- 学習終了後に何もしない
- 成果物保存後にPodをStopする
- 成果物保存後にPodをTerminateする

既定値は「何もしない」とする。Terminateは明示的に選択された場合のみ実行する。

自動Terminateの必須条件:

1. 学習プロセスの終了コードが0
2. 完成LoRAが存在し、0バイトではない
3. 必須成果物のexportディレクトリ作成に成功
4. rcloneの終了コードが0
5. Google Drive側のファイル一覧またはチェックサム検証に成功
6. 完了manifestをGoogle Driveへ保存済み
7. ユーザーが自動Terminateを有効化済み

上記のいずれかに失敗した場合はPodをTerminateせず、Gradio画面とログにエラーを表示する。

推奨処理順:

```text
学習完了
  → 成果物検証
  → export用ディレクトリ作成
  → Google Driveへrclone copy
  → 転送結果検証
  → 完了manifest保存
  → 30～60秒の猶予
  → RunPod APIでStopまたはTerminate
```

RunPod APIキーは環境変数として渡し、画面、ログ、設定ファイルへ出力しない。

### 22.8 rclone設定管理

- rclone remote名は既定で`gdrive`とするが設定変更可能にする
- `rclone.conf`をGitリポジトリへコミットしない
- Google認証情報はRunPodのSecretまたは環境変数、もしくは永続的に保護された設定ファイルとして渡す
- Gradio画面にアクセストークンやrefresh tokenを表示しない
- 接続テスト機能を設ける
- 学習開始前にGoogle Driveの空き容量と書き込み可否を可能な範囲で確認する
- 転送ログから秘密情報を除外する

---

## 23. ログ・障害対応

### 23.1 ログ分類

- application.log
- acquisition.log
- image_processing.log
- tagging.log
- training.log
- security.log

### 23.2 エラー表示

低レベルなスタックトレースだけでなく、ユーザー向けの原因候補を表示する。

例:

```text
CUDA out of memory
- batch sizeを2から1へ下げる
- gradient checkpointingを有効化する
- cache text encoder outputsを有効化する
- optimizerを8bit系へ変更する
- 他のGPUプロセスを停止する
```

### 23.3 ヘルスチェック

- DB接続
- Redis接続
- Worker応答
- GPU認識
- 書き込み権限
- 空き容量
- 学習バックエンド
- タガーモデル

---

## 24. バージョン管理

### 24.1 データセットスナップショット

学習開始時に以下を固定する。

- 採用画像ID
- 各画像のSHA-256
- 最終キャプション
- クロップ等の処理内容
- タグルール版
- データセット設定

画像選択やタグを変更しても、過去のTrainingRunの内容は変化しない。

### 24.2 設定マイグレーション

`settings_version`を持ち、ツール更新後に古いプロジェクトを読み込めること。

### 24.3 バックエンド固定

sd-scriptsのコミットIDまたはリリース版を保存し、自動更新で既存環境を壊さない。

---

## 25. テスト要件

### 25.1 単体テスト

- タグルール
- 重複判定
- パス検証
- 推奨パラメータ計算
- 総ステップ計算
- アダプター共通契約
- TOML生成
- ログ解析

### 25.2 結合テスト

- Danbooru検索から保存まで
- 取得中断・再開
- 画像破損処理
- タグ付けジョブ
- データセット生成
- 短時間の学習スモークテスト
- 学習停止・再開
- 成果物エクスポート

### 25.3 UIテスト

- 大量画像の仮想スクロール
- 複数選択
- フィルター維持
- 進捗更新
- 再接続
- エラー表示
- NSFWぼかし

---

## 26. 非機能要件

| 項目 | 要件 |
|---|---|
| 操作性 | 初心者モードと詳細モードを分離 |
| 性能 | 数千画像でも一覧操作が破綻しない |
| 可用性 | ブラウザ切断後もジョブ継続 |
| 復旧性 | 取得、タグ付け、学習を再開可能 |
| 拡張性 | Source、Tagger、Trainerをアダプター化 |
| 保守性 | UI、API、Worker、学習処理を分離 |
| 再現性 | 設定、版、seed、hashを保存 |
| 安全性 | 公開ポート、秘密情報、任意コマンドを制限 |
| 可観測性 | ログ、ジョブ状態、GPU、容量を確認可能 |
| 可搬性 | DockerでRunPod以外にも移行可能 |

---

## 27. 開発フェーズ案

### Phase 1: 最小実用版

- プロジェクト管理
- ローカル画像アップロード
- Danbooru取得
- SHA-256 / pHash重複判定
- 基本品質判定
- 画像選別UI
- WD Tagger
- タグ一括削除
- トリガーワード付与
- dataset TOML生成
- SDXL LoRA学習
- 手動設定
- ログと成果物表示

### Phase 2: 品質改善

- CLIP類似画像判定
- 高度な品質評価
- データセット診断
- 推奨パラメータ
- サンプル生成
- epoch比較
- state再開
- プロジェクトZIP
- モデルカード

### Phase 3: 拡張

- 追加取得元
- 外部ストレージ同期
- 複数学習バックエンド
- モデル系列別プリセット
- RunPod API連携
- 自動Pod停止
- ジョブキュー
- 自動パラメータ探索

---

## 28. 初期実装で優先すべき判断

1. **画像取得より先に、ローカル画像から一連の学習が完走する縦方向の機能を作る。**  
   Danbooru API固有の問題と学習処理の問題を分離できる。

2. **学習処理はsd-scriptsを直接呼び出し、独自学習コードを持たない。**  
   本ツールは設定生成、データ管理、実行監視、評価に注力する。

3. **自動除外は控えめにする。**  
   完全重複や破損以外は原則として「除外候補」とし、誤判定で良質な画像を失わない。

4. **推奨値は、最終的な正解ではなく根拠付きの初期値として扱う。**  
   学習元モデル、絵柄、画像の多様性によって最適値は変わる。

5. **画像とキャプションの版を固定してから学習する。**  
   学習後にタグを変更しても、過去の結果との対応が失われないようにする。

6. **RunPodのPodローカル領域と永続領域を明確に分ける。**  
   モデル、データセット、設定、成果物は永続領域へ保存する。

7. **最初からアダプター構造にする対象を限定する。**  
   少なくとも`ImageSourceAdapter`、`TaggerAdapter`、`TrainerAdapter`、`StorageAdapter`は分離する。

---

## 29. 完了条件

以下をすべて満たした時点を初期版の完成とする。

- Danbooruまたはアップロードから画像を登録できる
- 完全重複と近似重複を確認できる
- 自動除外候補と理由を一覧表示できる
- ユーザーが採用画像を選択できる
- 選択画像へ自動タグ付けできる
- 採用画像内のタグを出現回数順に表示し、チェックボックスで保持・削除を手動選択できる
- 全画像へトリガーワードを付与できる
- SDXL学習元モデルを指定できる
- 画像枚数とGPUに基づく推奨設定を作成できる
- 推奨設定を手動変更できる
- sd-scriptsによるLoRA学習を実行できる
- 学習進捗とエラーをUIで確認できる
- LoRA、設定、ログ、サンプルを保存できる
- ソースコードをGitリポジトリから取得・更新できる
- 学習元モデルをGoogle DriveからRunPodへ取得できる
- 学習途中のstate・checkpoint・cacheをRunPod内だけに保持できる
- 学習正常完了時に採用画像、最終キャプション、設定、完成LoRA、ログをGoogle Driveへ同期できる
- Google Drive同期に失敗した場合は自動Terminateしない
- 明示的に有効化した場合のみ、同期検証後にPodをStopまたはTerminateできる
- Podを再作成しても永続領域からプロジェクトを復元できる
- 取得元・タグ・学習設定を含む再現情報をエクスポートできる

---

## 30. 参考技術情報

- SDXL LoRA学習にはkohya-ss / sd-scriptsの`sdxl_train_network.py`を利用する構成を想定する。
- SDXLは2つのテキストエンコーダーを持つため、初期プリセットではUNetのみの学習を安全側の既定値とし、text encoder学習は詳細設定に置く。
- RunPodでは、永続化が必要なプロジェクトやモデルをNetwork Volume等の永続ストレージへ保存する。
- RunPod上ではWeb UI以外の内部サービス用ポートを不用意に公開しない。
- 自動タグ付けはWD 1.4系Taggerを初期候補とし、将来モデルを差し替えられるようアダプター化する。

参考:
- RunPod Documentation: Network Volumes / Pod Networking / Expose Ports
- kohya-ss / sd-scripts: SDXL LoRA Training Documentation
- SmilingWolf WD 1.4 Tagger model documentation
- Danbooru API documentationおよび利用規約

---

## 31. 推奨リポジトリ構成

```text
ldts/
├── app.py
├── ui/
│   ├── dashboard.py
│   ├── acquisition.py
│   ├── image_selection.py
│   ├── tag_editor.py
│   ├── training_settings.py
│   ├── training_monitor.py
│   └── artifacts.py
├── services/
│   ├── acquisition/
│   ├── deduplication/
│   ├── quality/
│   ├── tagging/
│   ├── datasets/
│   ├── recommendations/
│   ├── training/
│   ├── storage/
│   │   ├── local_workspace.py
│   │   └── rclone_gdrive.py
│   └── runpod/
│       └── lifecycle.py
├── adapters/
│   ├── sources/
│   ├── taggers/
│   ├── trainers/
│   └── storage/
├── jobs/
│   ├── manager.py
│   ├── state.py
│   └── subprocess_runner.py
├── database/
├── config/
├── scripts/
│   ├── start.sh
│   ├── setup_runpod.sh
│   └── verify_environment.sh
├── tests/
├── docs/
├── requirements.txt
├── pyproject.toml
├── .env.example
└── README.md
```

Gitにはソースコード、依存関係定義、設定例、DBマイグレーションだけを保存する。以下は`.gitignore`対象とする。

```text
.env
rclone.conf
*.safetensors
*.ckpt
/workspace/
/runtime/
/projects/
outputs/
cache/
logs/
```

---

## 32. 補足: 推奨パラメータ機能の実装方法

推奨パラメータ機能は、初期版では機械学習による自動探索ではなく、検証可能なルールエンジンとして実装する。

```python
RecommendationContext(
    concept_type="character",
    image_count=48,
    effective_image_count=36.5,
    median_resolution=(1400, 1800),
    duplicate_ratio=0.12,
    model_family="sdxl",
    gpu_vram_gb=24,
    priority="quality",
)
```

返却例:

```json
{
  "parameters": {
    "resolution": 1024,
    "enable_bucket": true,
    "network_dim": 32,
    "network_alpha": 16,
    "batch_size": 2,
    "gradient_checkpointing": true,
    "cache_latents": true,
    "mixed_precision": "bf16",
    "train_unet_only": true,
    "optimizer": "AdamW8bit",
    "unet_lr": 0.0001,
    "epochs": 10,
    "repeats": 5
  },
  "estimated_total_steps": 1200,
  "warnings": [],
  "reasons": {
    "network_dim": "画像枚数と概念種別から32を初期値とした",
    "batch_size": "24GB VRAMで1024学習を安全側に見積もった",
    "train_unet_only": "SDXLの初期安全設定"
  }
}
```

推奨ロジックには必ず版番号を付ける。

```text
recommendation_engine_version: 1.0.0
```

## Phase 7B: 学習中VRAM測定と校正互換性

学習開始時にはrecommendation provenanceとは独立したjob execution environment snapshotを作成する。snapshotは論理GPU index、nvidia-smiの物理GPU index、短縮GPU UUID fingerprint、architecture/compute capability、total VRAM、CUDA利用可否、正規化済み`CUDA_VISIBLE_DEVICES`、sd-scripts version、xformers可否、detector version、detected_atだけを保持し、immutableとする。手動設定や旧設定にも同じsnapshot経路を適用する。

`CUDA_VISIBLE_DEVICES`の論理GPUを物理index/UUIDへ解決し、PIDのprocess identity/groupとGPU UUIDを検証する。実行時snapshotのUUIDを測定・summary・校正の正本とし、recommendation時snapshotは差分比較だけに利用する。複数GPUで対象を一意に特定できない場合やGPUが途中で変わった場合は先頭GPUへフォールバックせず、`GPU_IDENTITY_UNAVAILABLE`、`EXPECTED_GPU_NOT_FOUND`、`TARGET_PID_NOT_FOUND`、`PROCESS_IDENTITY_MISMATCH`、`PROCESS_GROUP_MISMATCH`、`AMBIGUOUS_GPU_SELECTION`、`GPU_CHANGED_DURING_JOB`の固定コードを保存する。

学習workerはheartbeat、進捗解析、artifact走査から独立した間隔でGPUメモリを測定する。既定間隔は5秒で、`nvidia-smi`の固定queryと`shell=False`を使用する。ジョブPIDのprocess identity/groupと環境snapshotのGPU UUIDを確認できない測定はtarget process peakに採用せず、全GPUのfree/minimumを低信頼の参考値として扱う。

測定値はjobごとにboundedな集約レコードへ冪等に保存し、再起動後に復元する。終端性能サマリーは単発の終端測定ではなく集約レコードからtarget/全GPU/他プロセスpeak、開始前・最小・終了後free、件数、失敗件数、時刻範囲を読み取る。メモリ校正はtarget identity、サンプル数、confidence、coverage、他プロセス影響、GPU不変性を検証する。

校正の速度・VRAMグループはGPU identity/VRAM class/architecture、resolution、batch、gradient accumulation/effective batch、LoRA module/dim/alpha、optimizer、precision、cache/checkpointing、world size、sd-scripts version、xformers可否を明示的に比較する。OOM補正は同等条件のmedium以上の履歴だけでbatch低下を提案し、低信頼・異なるbatch/dim/GPUの履歴はwarningに留める。summary content fingerprintとcalibration state fingerprintを分離し、履歴の採用、除外、理由、再分類、再収集、集約更新は関連校正をstaleにする。適用直前にもsnapshotと現在の推奨・GPUの互換性を再検証する。

これにより、ツール更新前後で推奨値が変わった理由を追跡できる。
## Phase 2B実装補足

Phase 2Bでは、ImageHashのpHashをEXIF Orientation補正、透過画像の白背景合成、RGB正規化後に計算する。pHashは固定長16進文字列としてアルゴリズム、hash_size、detector_version、計算状態、日時とともに保存する。同一設定のハミング距離が閾値以下の無向グラフ連結成分を類似グループとし、完全重複のみのグループは`exact_only`として区別する。手動代表と手動否定ペアは自動再検査で保護し、原画像、既存のSHA-256／品質検査、SelectionStateは変更しない。
