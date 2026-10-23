# deep-learning-from-scratch-6

my environment for https://github.com/oreilly-japan/deep-learning-from-scratch-6

## 今後の環境移行方針

eMudhra 対応のため、Devbox／Nix と Python 3.12 への依存を外し、uv と Python 3.15 系へ移行する。
Devbox／Nix は削除済み。Python 3.15 系への移行は未実施。

## セットアップ

[uv](https://docs.astral.sh/uv/) を事前にインストールしておく。

```bash
# submodule の取得（clone 直後のみ）
git submodule update --init

# Python パッケージのインストール（uv.lock に固定されたバージョンを original/.venv/ に入れる）
uv sync --project original --all-extras --frozen
```

`--frozen` により `uv.lock` は書き換えられない。

学習済みモデルが必要な章では、事前にダウンロードしておく（Hugging Face から数百 MB 級のファイルを取得する）。
スクリプトは wget を使うので、未インストールなら `brew install wget` で入れておく。

```bash
bash original/codebot/download.sh
bash original/storybot/download.sh
bash original/webbot/download.sh
```

## 実行方法

公式のスクリプトはリポジトリルート（`original/`）か各章のフォルダから実行する想定なので、まず `original/` に移動して `.venv` を有効化する。

```bash
cd original
source .venv/bin/activate

# 親フォルダから実行
python ch01/01_char_tokenizer.py

# 各章のフォルダ内で実行
cd ch01
python 01_char_tokenizer.py
```

venv を有効化せず 1 コマンドで実行する場合:

```bash
uv run --project original --directory original python ch01/01_char_tokenizer.py
```
