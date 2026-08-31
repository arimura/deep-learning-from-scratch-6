// Package bpetokenizer は BPE (Byte Pair Encoding) のトークナイザを提供する。
//
// bpetrain.Train で学習したマージ規則と end token を受け取り、トークン ID と
// バイト列の対応表を構築する。0〜255 のトークン ID は対応するバイトそのもの、
// 256 以降のトークン ID はマージ規則に従って 2 つのトークンのバイト列を
// 連結したものであり、語彙の最後の 1 語は end token に割り当てられる。
package bpetokenizer

import (
	"fmt"
	"strings"
	"unicode/utf8"

	"github.com/arimura/deep-learning-from-scratch-6/dev/ch01/bpetrain"
	"github.com/arimura/deep-learning-from-scratch-6/dev/ch01/bytetokenizer"
)

// Tokenizer は BPE トークナイザ。
type Tokenizer struct {
	// merges は学習で得られたマージ規則を適用順に並べたもの。
	merges []bpetrain.MergeRule
	// endToken は特殊トークンとして扱う end token の文字列。
	endToken string
	// endTokenID は end token に割り当てたトークン ID。
	endTokenID int
	// idToBytes はトークン ID からバイト列への対応表。
	idToBytes map[int][]byte
	// vocabSize は語彙サイズ (トークン ID の総数)。
	vocabSize int
}

// New はマージ規則から Tokenizer を生成する。
// end token には bpetrain.DefaultSpecialToken を用いる。詳細は NewWithEndToken を参照。
func New(merges []bpetrain.MergeRule) (*Tokenizer, error) {
	return NewWithEndToken(merges, bpetrain.DefaultSpecialToken)
}

// NewWithEndToken はマージ規則と end token から Tokenizer を生成する。
//
// ID からバイト列への対応表は次のように構築される。
//   - 0〜255: バイト値そのもの ([]byte{id})
//   - 256 〜 256+len(merges)-1: マージ規則の Pair が表す 2 つのバイト列を連結したもの
//   - 256+len(merges): end token の文字列のバイト列
//
// 語彙サイズは bytetokenizer.VocabSize (256) + len(merges) + 1 (end token 分) となる。
//
// マージ規則は bpetrain.Train が返す形式、すなわち NewID が 256 から連番で
// 並び、Pair の各要素がその時点で既に対応表に存在する ID であることを前提とする。
// この前提を満たさない場合、または endToken が空の場合はエラーを返す。
func NewWithEndToken(merges []bpetrain.MergeRule, endToken string) (*Tokenizer, error) {
	if endToken == "" {
		return nil, fmt.Errorf("bpetokenizer: end token must not be empty")
	}

	idToBytes := make(map[int][]byte, bytetokenizer.VocabSize+len(merges)+1)
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

	endTokenID := bytetokenizer.VocabSize + len(merges)
	idToBytes[endTokenID] = []byte(endToken)

	copied := make([]bpetrain.MergeRule, len(merges))
	copy(copied, merges)

	return &Tokenizer{
		merges:     copied,
		endToken:   endToken,
		endTokenID: endTokenID,
		idToBytes:  idToBytes,
		vocabSize:  endTokenID + 1,
	}, nil
}

// EndToken は end token の文字列を返す。
func (t *Tokenizer) EndToken() string {
	return t.endToken
}

// EndTokenID は end token に割り当てたトークン ID を返す。
func (t *Tokenizer) EndTokenID() int {
	return t.endTokenID
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

// Encode は文字列を BPE のトークン ID 列に変換する。
//
// まず文字列を end token で分割し、end token は対応するトークン ID に、
// それ以外の部分は encodeText でトークン ID 列に変換して連結する。
// 不正な UTF-8 バイト列が含まれる場合はエラーを返す。
func (t *Tokenizer) Encode(s string) ([]int, error) {
	ids := []int{}
	for _, part := range t.splitByEndToken(s) {
		if part == t.endToken {
			ids = append(ids, t.endTokenID)
			continue
		}
		partIDs, err := t.encodeText(part)
		if err != nil {
			return nil, err
		}
		ids = append(ids, partIDs...)
	}
	return ids, nil
}

// splitByEndToken は s を end token で分割した配列を返す。
// end token 自身も 1 つの要素として結果に含まれる。
// 例: end token が "<e>" のとき "a<e>b" は ["a", "<e>", "b"] となる。
func (t *Tokenizer) splitByEndToken(s string) []string {
	var parts []string
	for {
		i := strings.Index(s, t.endToken)
		if i < 0 {
			return append(parts, s)
		}
		parts = append(parts, s[:i], t.endToken)
		s = s[i+len(t.endToken):]
	}
}

// encodeText は end token を含まない文字列をトークン ID 列に変換する。
//
// まず文字列を UTF-8 バイト列 (トークン ID 0〜255) に変換し、
// マージ規則を学習された順に 1 つずつ適用して隣接ペアを新しいトークン ID に
// 置き換える。マージ規則の適用には bpetrain.Merge を用いる。
// 不正な UTF-8 バイト列が含まれる場合はエラーを返す。
func (t *Tokenizer) encodeText(s string) ([]int, error) {
	ids, err := bytetokenizer.New().Encode(s)
	if err != nil {
		return nil, fmt.Errorf("bpetokenizer: %w", err)
	}
	for _, m := range t.merges {
		if len(ids) < 2 {
			break
		}
		ids = bpetrain.Merge(ids, m.Pair, m.NewID)
	}
	return ids, nil
}

// Decode はトークン ID 列を文字列に変換する。
//
// 各トークン ID を対応表でバイト列に展開し、すべて連結したものを
// UTF-8 の文字列として返す。対応表に存在しないトークン ID が含まれる場合、
// または連結した結果が正しい UTF-8 バイト列にならない場合はエラーを返す。
func (t *Tokenizer) Decode(ids []int) (string, error) {
	buf := make([]byte, 0, len(ids))
	for i, id := range ids {
		b, ok := t.idToBytes[id]
		if !ok {
			return "", fmt.Errorf("bpetokenizer: unknown token id %d at index %d", id, i)
		}
		buf = append(buf, b...)
	}
	if !utf8.Valid(buf) {
		return "", fmt.Errorf("bpetokenizer: decoded bytes are not valid UTF-8")
	}
	return string(buf), nil
}

// Merges は学習で得られたマージ規則のコピーを適用順に返す。
func (t *Tokenizer) Merges() []bpetrain.MergeRule {
	merges := make([]bpetrain.MergeRule, len(t.merges))
	copy(merges, t.merges)
	return merges
}
