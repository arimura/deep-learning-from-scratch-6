// Package bpetrain は BPE (Byte Pair Encoding) の学習に用いる基本処理を提供する。
//
// BPE の学習は、トークン ID 列の中で最も頻出する隣接ペアを新しいトークン ID に
// 置き換える操作の繰り返しである。本パッケージはその 1 ステップを構成する
// 「隣接ペアの出現回数のカウント」と「指定ペアの置き換え」を提供する。
package bpetrain

import (
	"fmt"

	"github.com/arimura/deep-learning-from-scratch-6/dev/ch01/bytetokenizer"
)

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

// MergeRule は学習によって得られたマージ規則を表す。
// Pair を NewID に置き換えることを意味し、学習された順序 (ID の昇順) に並ぶ。
type MergeRule struct {
	Pair  Pair
	NewID int
}

// Train は text を対象に BPE の学習を行う。
//
// 初期のトークン ID は UTF-8 バイト値 (0〜255) であり、初期語彙サイズは
// bytetokenizer.VocabSize (256) である。vocabSize との差がマージ回数となり、
// 新しいトークン ID は 256 から順に連番で割り当てられる。
//
// 戻り値は学習後のトークン ID 列と、適用した順に並んだマージ規則である。
// マージできる隣接ペアが無くなった場合は、マージ回数に達していなくても
// そこで学習を打ち切る。
//
// vocabSize が初期語彙サイズ未満の場合、または text が不正な UTF-8 の場合は
// エラーを返す。
func Train(text string, vocabSize int) ([]int, []MergeRule, error) {
	if vocabSize < bytetokenizer.VocabSize {
		return nil, nil, fmt.Errorf("bpetrain: vocabSize %d must be at least %d", vocabSize, bytetokenizer.VocabSize)
	}

	ids, err := bytetokenizer.New().Encode(text)
	if err != nil {
		return nil, nil, fmt.Errorf("bpetrain: %w", err)
	}

	numMerges := vocabSize - bytetokenizer.VocabSize
	merges := make([]MergeRule, 0, numMerges)
	for i := 0; i < numMerges; i++ {
		counts := CountPairs(ids)
		best, ok := mostFrequentPair(ids, counts)
		if !ok {
			break
		}
		newID := bytetokenizer.VocabSize + i
		ids = Merge(ids, best, newID)
		merges = append(merges, MergeRule{Pair: best, NewID: newID})
	}
	return ids, merges, nil
}

// mostFrequentPair は counts の中で最も出現回数の多いペアを返す。
// 同数のペアが複数ある場合は ids の中で先に現れたものを選び、
// 結果が map の反復順に依存しないようにする。
// マージ対象のペアが無い場合は第 2 戻り値に false を返す。
func mostFrequentPair(ids []int, counts map[Pair]int) (Pair, bool) {
	var best Pair
	bestCount := 0
	for i := 0; i+1 < len(ids); i++ {
		p := Pair{First: ids[i], Second: ids[i+1]}
		if c := counts[p]; c > bestCount {
			best, bestCount = p, c
		}
	}
	return best, bestCount > 0
}
