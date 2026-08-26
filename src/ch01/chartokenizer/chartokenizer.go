// Package chartokenizer は文字単位のトークナイザを提供する。
// 各文字 (rune) を Unicode コードポイントの整数値としてエンコードし、
// 整数値の列から元の文字列へデコードする。
package chartokenizer

import (
	"fmt"
	"unicode/utf8"
)

// Tokenizer は文字単位のトークナイザ。
// トークン ID は文字の Unicode コードポイントそのものである。
type Tokenizer struct{}

// New は Tokenizer を生成する。
func New() *Tokenizer {
	return &Tokenizer{}
}

// Encode は文字列を Unicode コードポイントの列に変換する。
// 不正な UTF-8 バイト列が含まれる場合はエラーを返す。
func (t *Tokenizer) Encode(s string) ([]int, error) {
	if !utf8.ValidString(s) {
		return nil, fmt.Errorf("chartokenizer: invalid UTF-8 string")
	}
	ids := make([]int, 0, utf8.RuneCountInString(s))
	for _, r := range s {
		ids = append(ids, int(r))
	}
	return ids, nil
}

// Decode は Unicode コードポイントの列を文字列に変換する。
// 不正なコードポイント (負数、サロゲート、上限超過) が含まれる場合はエラーを返す。
func (t *Tokenizer) Decode(ids []int) (string, error) {
	buf := make([]byte, 0, len(ids))
	for i, id := range ids {
		r := rune(id)
		if id < 0 || id > utf8.MaxRune || !utf8.ValidRune(r) {
			return "", fmt.Errorf("chartokenizer: invalid code point %d at index %d", id, i)
		}
		buf = utf8.AppendRune(buf, r)
	}
	return string(buf), nil
}
