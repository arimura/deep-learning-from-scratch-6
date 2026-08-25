# deep-learning-from-scratch-6

my environment for https://github.com/oreilly-japan/deep-learning-from-scratch-6

## セットアップ

```bash
# submodule の取得（clone 直後のみ）
git submodule update --init

# Python パッケージのインストール（uv.lock に固定されたバージョンを .venv/ に入れる）
devbox run setup
```

`setup` は `uv sync --project original --all-extras --frozen` を実行する。
`--frozen` により `uv.lock` は書き換えられない。

学習済みモデルが必要な章では、事前にダウンロードしておく（Hugging Face から数百 MB 級のファイルを取得する）。

```bash
devbox run download
```

## 実行方法

devbox shell に入ると `.venv` が自動で有効化されるので、公式 README と同じ手順で実行できる。
公式のスクリプトはリポジトリルート（`original/`）か各章のフォルダから実行する想定なので、まず `original/` に移動する。

```bash
devbox shell
cd original

# 親フォルダから実行
python ch01/01_char_tokenizer.py

# 各章のフォルダ内で実行
cd ch01
python 01_char_tokenizer.py
```

shell に入らず 1 コマンドで実行する場合:

```bash
devbox run -- uv run --project original --directory original python ch01/01_char_tokenizer.py
```
