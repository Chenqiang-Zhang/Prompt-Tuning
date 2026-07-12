# Personalized Dialogue with Prompt-Tuning — 再現実装

Kasahara et al. (2022) *"Building a Personalized Dialogue System with Prompt-Tuning"*
(NAACL SRW) のスクラッチ再現。ローカル（Apple Silicon / MPS）で LLM を動かし、
prompt-tuning の全工程を学ぶことを目的とする。

- 論文: https://aclanthology.org/2022.naacl-srw.13.pdf
- 言語トラック: **日本語**（自動評価のみ、手動評価は省略）
- ベースLM: **Qwen2.5**（`0.5B` → `1.5B`/`3B`/`7B` にスケール可能）

## 手法（論文の核）

事前学習済み Causal LM の**全パラメータ（トークン埋め込み層を含む）を凍結**し、
長さ `L`（論文=200）の**新しい学習可能な埋め込み行列＝「persona info tokens」**だけを追加する。
毎回の forward でこのソフトプロンプトを入力埋め込みの先頭に連結する:

```
[ soft_prompt (L,H) ]  ++  embed( 発話 ++ 区切り ++ 応答 )   ->  凍結LM
        ↑学習対象                        ↑凍結埋め込み層
```

- 損失は**応答トークンのみ**（`soft_prompt` と発話部分は label=-100 でマスク）
- `soft_prompt` は **persona 文の埋め込みで初期化**（L トークン埋まるまで繰り返し）
- 学習は Adam, lr=1e-3（論文 Appendix B）／推論は **greedy**
- Lester et al. (2021) の prompt-tuning を、persona 初期化つきで対話に適用したもの

実装は [`src/soft_prompt.py`](src/soft_prompt.py) の `SoftPromptDialogue` に集約。
学習対象は soft prompt のみ（0.5B で全体の **0.036%**、200×896=179,200 パラメータ）。

## データ

論文の JPersonaChat はライセンス申請が必要なため、構造が同じで自由入手できる
**RealPersonaChat**（`nu-dialogue/real-persona-chat`, 話者ごとの persona 10文 +
2者マルチターン対話）で代替する。

`src/prepare_data.py` が論文のレシピを再現:
1. マルチターン対話を「対話ペア」= (直前の発話, 応答) に分割
2. **応答者の persona** ごとにペアを集約
3. ペア数上位 N ペルソナ（論文=3）を選択、各を **9:1** で train/eval 分割
4. persona と無関係な**短い一般ペア**を混合（論文が DailyDialog/JEmpatheticDialogues を
   混ぜるのに対応。ここでは他ペルソナの短い対話を流用し完全に無料・自己完結）

paper に合わせ各ペルソナ 525 ペアにキャップ（`--max_pairs_per_persona 525`）。

## セットアップ

### クラウド（CUDA GPU）— 推奨

```bash
git clone <this-repo-url> && cd Prompt-Tuning
bash scripts/setup_cloud.sh          # 依存導入 + データDL + 整形を一括
# 実験（CUDA では bfloat16 推奨）
scripts/run_experiment.sh Qwen/Qwen2.5-3B 15 qwen3b bfloat16
```

`setup_cloud.sh` は GPU 検出 → `pip install -r requirements.txt`（CUDA ホストでは
既定の torch wheel が CUDA 対応）→ RealPersonaChat DL → `prepare_data.py` を実行する。
`--device auto` が CUDA を自動選択する。

### ローカル（手動）

```bash
conda create -y -n ptune python=3.11 && conda activate ptune
pip install -r requirements.txt
mkdir -p data
curl -sL -o data/rpc.zip \
  https://github.com/nu-dialogue/real-persona-chat/archive/refs/tags/v1.0.0.zip
(cd data && unzip -q -o rpc.zip)
```

## 実行

```bash
# 1) データ整形（上位3ペルソナ、各525ペア、一般ペア 1:1 混合）
python src/prepare_data.py --top_personas 3 --general_ratio 1.0 --max_pairs_per_persona 525

# 2) 学習（persona ごと）— 例: 0.5B, 15 epochs
python src/train.py --persona_dir data/processed/persona_CP \
  --model Qwen/Qwen2.5-0.5B --out outputs/qwen05b/persona_CP.pt --epochs 15

# 3) 生成（persona-eval / general-eval）
python src/generate.py --model Qwen/Qwen2.5-0.5B --prompt outputs/qwen05b/persona_CP.pt \
  --eval_file data/processed/persona_CP/eval_persona.jsonl --out outputs/qwen05b/gen_persona_CP.jsonl

# 4) 自動評価 distinct-1/2
python src/eval_distinct.py outputs/qwen05b/gen_persona_*.jsonl

# 全部まとめて（3ペルソナ学習→生成→distinct）
scripts/run_experiment.sh Qwen/Qwen2.5-0.5B 15 qwen05b

# 対話を試す
python src/generate.py --model Qwen/Qwen2.5-0.5B --prompt outputs/qwen05b/persona_CP.pt --interactive
```

大きいモデルは bf16 推奨: `--dtype bfloat16`（例 `Qwen/Qwen2.5-3B`）。

## ファイル

| ファイル | 役割 |
|---|---|
| `src/prepare_data.py` | RealPersonaChat → persona 別対話ペア + 一般ペア混合、9:1分割 |
| `src/soft_prompt.py`  | **★スクラッチの soft prompt 層**（追加埋め込み層・応答のみ損失・persona初期化） |
| `src/train.py`        | prompt-tuning 学習（soft prompt のみ Adam 更新） |
| `src/generate.py`     | greedy 生成（`inputs_embeds` 経由）／対話モード |
| `src/eval_distinct.py`| distinct-1/2（多様性、Table 1/6 の指標） |
| `scripts/run_experiment.sh` | 3ペルソナ一括: 学習→生成→評価 |

## 論文との対応と差分

| 項目 | 論文 | 本実装 |
|---|---|---|
| 手法 | 凍結LM + 学習可能 persona 埋め込み層(L=200) | 同じ（スクラッチ） |
| 損失 | 応答トークンのみ | 同じ |
| 初期化 | persona 文の埋め込みを繰り返し | 同じ |
| 最適化 | Adam, lr=1e-3, greedy | 同じ |
| LM | GPT-2 / GPT-J-6B / HyperCLOVA | **Qwen2.5**（ローカル都合） |
| 日本語データ | JPersonaChat + JEmpatheticDialogues | **RealPersonaChat**（無料・同構造） |
| 一般ペアの区切り | GPT系は連結のみ | 軽量に改行 `\n` を挿入（Qwen に turn token が無いため） |
| 評価 | distinct + 人手(MTurk) | **distinct のみ**（人手は省略、生成例で定性確認） |

distinct は日本語では文字 n-gram で算出（`--word` で単語分割も可）。
