// Package bpetrain は BPE (Byte Pair Encoding) の学習に用いる基本処理を提供する。
//
// BPE の学習は、トークン ID 列の中で最も頻出する隣接ペアを新しいトークン ID に
// 置き換える操作の繰り返しである。本パッケージはその 1 ステップを構成する
// 「隣接ペアの出現回数のカウント」と「指定ペアの置き換え」を提供する。
package bpetrain

import (
	"fmt"
	"strings"

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
// counts が nil でない場合は、その map に出現回数を加算して同じ map を返す。
// これにより複数のトークン ID 列にわたる出現回数を 1 つの map に集約できる。
// counts が nil の場合は新しい map を作成して返す。
//
// 出現回数は左から右へ、重なりを許さずに数えるのではなく、
// すべての隣接位置 (i, i+1) を独立に数える。
// したがって [1 1 1] の場合、ペア {1, 1} の出現回数は 2 となる。
//
// ids の長さが 1 以下の場合、counts はそのまま (nil なら空の map を) 返す。
func CountPairs(ids []int, counts map[Pair]int) map[Pair]int {
	if counts == nil {
		counts = make(map[Pair]int)
	}
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

// DefaultSpecialToken は Train が既定で用いる特殊トークン。
const DefaultSpecialToken = "<|endoftext|>"

// MinVocabSize は Train に指定できる最小のボキャブラリーサイズ。
// 初期語彙 (0〜255 のバイト値) に特殊トークン用の 1 語を加えたものである。
const MinVocabSize = bytetokenizer.VocabSize + 1

// Train は text を対象に BPE の学習を行う。
// 特殊トークンには DefaultSpecialToken を用いる。詳細は TrainWithSpecialToken を参照。
func Train(text string, vocabSize int) ([][]int, []MergeRule, error) {
	return TrainWithSpecialToken(text, vocabSize, DefaultSpecialToken)
}

// TrainWithSpecialToken は text を対象に BPE の学習を行う。
//
// text は specialToken で分割され、分割された各文字列を UTF-8 バイト値 (0〜255) の
// トークン ID 列に変換したもの (トークン ID 列の配列) を学習対象とする。
// 特殊トークンをまたぐ隣接ペアはカウントもマージもされない。
//
// 初期語彙サイズは bytetokenizer.VocabSize (256) であり、これに特殊トークン用の
// 1 語を加えた MinVocabSize と vocabSize との差がマージ回数となる。
// 新しいトークン ID は 256 から順に連番で割り当てられ、特殊トークンには
// 語彙の最後の 1 語分が予約される。
//
// 各マージでは、全トークン ID 列の隣接ペアの出現回数を 1 つのハッシュテーブルに
// 集約し、最頻出のペアを新しいトークン ID に置き換える。置き換えは全トークン
// ID 列に対して行う。マージできる隣接ペアが無くなった場合は、マージ回数に
// 達していなくてもそこで学習を打ち切る。
//
// 戻り値は学習後のトークン ID 列の配列 (specialToken による分割と同じ順序) と、
// 適用した順に並んだマージ規則である。
//
// vocabSize が MinVocabSize 未満の場合、specialToken が空の場合、
// または text が不正な UTF-8 の場合はエラーを返す。
func TrainWithSpecialToken(text string, vocabSize int, specialToken string) ([][]int, []MergeRule, error) {
	if vocabSize < MinVocabSize {
		return nil, nil, fmt.Errorf("bpetrain: vocabSize %d must be at least %d", vocabSize, MinVocabSize)
	}
	if specialToken == "" {
		return nil, nil, fmt.Errorf("bpetrain: special token must not be empty")
	}

	chunks, err := splitToIDs(text, specialToken)
	if err != nil {
		return nil, nil, err
	}

	numMerges := vocabSize - MinVocabSize
	merges := make([]MergeRule, 0, numMerges)
	for i := 0; i < numMerges; i++ {
		counts := make(map[Pair]int)
		for _, ids := range chunks {
			counts = CountPairs(ids, counts)
		}
		if len(counts) == 0 {
			break
		}
		best := mostFrequentPair(chunks, counts)
		newID := bytetokenizer.VocabSize + i
		for j, ids := range chunks {
			chunks[j] = Merge(ids, best, newID)
		}
		merges = append(merges, MergeRule{Pair: best, NewID: newID})
	}
	return chunks, merges, nil
}

// splitToIDs は text を specialToken で分割し、各部分をバイト値のトークン ID 列に変換する。
func splitToIDs(text, specialToken string) ([][]int, error) {
	parts := strings.Split(text, specialToken)
	enc := bytetokenizer.New()
	chunks := make([][]int, 0, len(parts))
	for _, p := range parts {
		ids, err := enc.Encode(p)
		if err != nil {
			return nil, fmt.Errorf("bpetrain: %w", err)
		}
		chunks = append(chunks, ids)
	}
	return chunks, nil
}

// mostFrequentPair は counts の中で最も出現回数の多いペアを返す。
// 同数のペアが複数ある場合は chunks の中で先に現れたものを選び、
// 結果が map の反復順に依存しないようにする。
// counts は空であってはならない。
func mostFrequentPair(chunks [][]int, counts map[Pair]int) Pair {
	var best Pair
	bestCount := 0
	for _, ids := range chunks {
		for i := 0; i+1 < len(ids); i++ {
			p := Pair{First: ids[i], Second: ids[i+1]}
			if c := counts[p]; c > bestCount {
				best, bestCount = p, c
			}
		}
	}
	return best
}
