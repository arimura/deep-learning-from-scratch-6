// Package bpetokenizer は BPE (Byte Pair Encoding) のトークナイザを提供する。
//
// bpetrain.Train で学習したマージ規則を受け取り、トークン ID とバイト列の
// 対応表を構築する。0〜255 のトークン ID は対応するバイトそのもの、
// 256 以降のトークン ID はマージ規則に従って 2 つのトークンのバイト列を
// 連結したものである。
package bpetokenizer

import (
	"fmt"

	"github.com/arimura/deep-learning-from-scratch-6/dev/ch01/bpetrain"
	"github.com/arimura/deep-learning-from-scratch-6/dev/ch01/bytetokenizer"
)

// Tokenizer は BPE トークナイザ。
type Tokenizer struct {
	// merges は学習で得られたマージ規則を適用順に並べたもの。
	merges []bpetrain.MergeRule
	// idToBytes はトークン ID からバイト列への対応表。
	idToBytes map[int][]byte
	// vocabSize は語彙サイズ (トークン ID の総数)。
	vocabSize int
}

// New はマージ規則から Tokenizer を生成する。
//
// ID からバイト列への対応表は次のように構築される。
//   - 0〜255: バイト値そのもの ([]byte{id})
//   - 256 以降: マージ規則の Pair が表す 2 つのバイト列を連結したもの
//
// 語彙サイズは bytetokenizer.VocabSize (256) + len(merges) となる。
//
// マージ規則は bpetrain.Train が返す形式、すなわち NewID が 256 から連番で
// 並び、Pair の各要素がその時点で既に対応表に存在する ID であることを前提とする。
// この前提を満たさない場合はエラーを返す。
func New(merges []bpetrain.MergeRule) (*Tokenizer, error) {
	idToBytes := make(map[int][]byte, bytetokenizer.VocabSize+len(merges))
	for i := 0; i < bytetokenizer.VocabSize; i++ {
		idToBytes[i] = []byte{byte(i)}
	}

	for i, m := range merges {
		if want := bytetokenizer.VocabSize + i; m.NewID != want {
			return nil, fmt.Errorf("bpetokenizer: merges[%d].NewID = %d, want %d", i, m.NewID, want)
		}
		first, ok := idToBytes[m.Pair.First]
		if !ok {
			return nil, fmt.Errorf("bpetokenizer: merges[%d].Pair.First: unknown token id %d", i, m.Pair.First)
		}
		second, ok := idToBytes[m.Pair.Second]
		if !ok {
			return nil, fmt.Errorf("bpetokenizer: merges[%d].Pair.Second: unknown token id %d", i, m.Pair.Second)
		}
		b := make([]byte, 0, len(first)+len(second))
		b = append(b, first...)
		b = append(b, second...)
		idToBytes[m.NewID] = b
	}

	copied := make([]bpetrain.MergeRule, len(merges))
	copy(copied, merges)

	return &Tokenizer{
		merges:    copied,
		idToBytes: idToBytes,
		vocabSize: bytetokenizer.VocabSize + len(merges),
	}, nil
}

// VocabSize は語彙サイズ (トークン ID の総数) を返す。
func (t *Tokenizer) VocabSize() int {
	return t.vocabSize
}

// Bytes はトークン ID に対応するバイト列のコピーを返す。
// id が対応表に存在しない場合は第 2 戻り値に false を返す。
func (t *Tokenizer) Bytes(id int) ([]byte, bool) {
	src, ok := t.idToBytes[id]
	if !ok {
		return nil, false
	}
	b := make([]byte, len(src))
	copy(b, src)
	return b, true
}

// Merges は学習で得られたマージ規則のコピーを適用順に返す。
func (t *Tokenizer) Merges() []bpetrain.MergeRule {
	merges := make([]bpetrain.MergeRule, len(t.merges))
	copy(merges, t.merges)
	return merges
}
