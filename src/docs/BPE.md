# Byte Pair Encoding

## BPE train
BPE学習用モジュールは以下の処理から構成される

- count pairs
  - 引数のトークンID配列で隣接するidのペアの出現回数をカウントする
  - dictionaryとして結果を返す
- merge
  - 指定されたペアを新しいトークンIDに置き換える
  - 引数にはトークンID配列、置き換えるペア、新しいトークンID、をとる
  - mergeされたトークンIDの配列を返す
