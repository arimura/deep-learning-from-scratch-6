// Package bpetrain は BPE (Byte Pair Encoding) の学習に用いる基本処理を提供する。
//
// BPE の学習は、トークン ID 列の中で最も頻出する隣接ペアを新しいトークン ID に
// 置き換える操作の繰り返しである。本パッケージはその 1 ステップを構成する
// 「隣接ペアの出現回数のカウント」と「指定ペアの置き換え」を提供する。
package bpetrain

// Pair は隣接する 2 つのトークン ID の組を表す。
// First が前方、Second が後方のトークン ID である。
type Pair struct {
	First  int
	Second int
}

// CountPairs は ids の中で隣接するトークン ID のペアの出現回数を数える。
//
// 出現回数は左から右へ、重なりを許さずに数えるのではなく、
// すべての隣接位置 (i, i+1) を独立に数える。
// したがって [1 1 1] の場合、ペア {1, 1} の出現回数は 2 となる。
//
// ids の長さが 1 以下の場合は空の map を返す。
func CountPairs(ids []int) map[Pair]int {
	counts := make(map[Pair]int)
	for i := 0; i+1 < len(ids); i++ {
		counts[Pair{First: ids[i], Second: ids[i+1]}]++
	}
	return counts
}

// Merge は ids の中の pair に一致する隣接ペアをすべて newID に置き換えた
// 新しいスライスを返す。引数の ids は変更しない。
//
// 置き換えは左から右へ貪欲に行い、一度置き換えた要素は次のペアの
// 前方要素としては扱わない。したがって [1 1 1] のペア {1, 1} を 9 に
// 置き換えると [9 1] となる。
func Merge(ids []int, pair Pair, newID int) []int {
	merged := make([]int, 0, len(ids))
	for i := 0; i < len(ids); {
		if i+1 < len(ids) && ids[i] == pair.First && ids[i+1] == pair.Second {
			merged = append(merged, newID)
			i += 2
			continue
		}
		merged = append(merged, ids[i])
		i++
	}
	return merged
}
