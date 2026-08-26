// Package bytetokenizer はバイト単位のトークナイザを提供する。
// 文字列を UTF-8 でエンコードしたバイト列 (各バイトは 0〜255 の整数) としてエンコードし、
// 整数値の列から元の文字列へデコードする。
//
// 文字 (Unicode コードポイント) は UTF-8 では 1〜4 バイトで表現されるため、
// 1 文字が複数のトークンに分かれることがある。語彙サイズは常に 256 で固定である。
package bytetokenizer

import (
	"fmt"
	"unicode/utf8"
)

// VocabSize はトークナイザの語彙サイズ (0〜255 のバイト値)。
const VocabSize = 256

// Tokenizer はバイト単位のトークナイザ。
// トークン ID は UTF-8 バイト列の各バイト値そのものである。
type Tokenizer struct{}

// New は Tokenizer を生成する。
func New() *Tokenizer {
	return &Tokenizer{}
}

// Encode は文字列を UTF-8 バイト列 (各バイトを整数化したもの) に変換する。
// 不正な UTF-8 バイト列が含まれる場合はエラーを返す。
func (t *Tokenizer) Encode(s string) ([]int, error) {
	if !utf8.ValidString(s) {
		return nil, fmt.Errorf("bytetokenizer: invalid UTF-8 string")
	}
	ids := make([]int, 0, len(s))
	for i := 0; i < len(s); i++ {
		ids = append(ids, int(s[i]))
	}
	return ids, nil
}

// Decode はバイト値の列を文字列に変換する。
// 0〜255 の範囲外の値が含まれる場合、または結合した結果が
// 正しい UTF-8 バイト列にならない場合はエラーを返す。
func (t *Tokenizer) Decode(ids []int) (string, error) {
	buf := make([]byte, 0, len(ids))
	for i, id := range ids {
		if id < 0 || id >= VocabSize {
			return "", fmt.Errorf("bytetokenizer: invalid byte value %d at index %d", id, i)
		}
		buf = append(buf, byte(id))
	}
	if !utf8.Valid(buf) {
		return "", fmt.Errorf("bytetokenizer: decoded bytes are not valid UTF-8")
	}
	return string(buf), nil
}
