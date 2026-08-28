package bpetokenizer

import (
	"reflect"
	"testing"

	"github.com/arimura/deep-learning-from-scratch-6/dev/ch01/bpetrain"
	"github.com/arimura/deep-learning-from-scratch-6/dev/ch01/bytetokenizer"
)

func TestNewNoMerges(t *testing.T) {
	tok, err := New(nil)
	if err != nil {
		t.Fatalf("New(nil) returned error: %v", err)
	}
	if got := tok.VocabSize(); got != bytetokenizer.VocabSize {
		t.Errorf("VocabSize() = %d, want %d", got, bytetokenizer.VocabSize)
	}
	for id := 0; id < bytetokenizer.VocabSize; id++ {
		b, ok := tok.Bytes(id)
		if !ok {
			t.Fatalf("Bytes(%d) = not found", id)
		}
		if want := []byte{byte(id)}; !reflect.DeepEqual(b, want) {
			t.Errorf("Bytes(%d) = %v, want %v", id, b, want)
		}
	}
	if got := tok.Merges(); len(got) != 0 {
		t.Errorf("Merges() = %v, want empty", got)
	}
}

func TestNewWithMerges(t *testing.T) {
	// "abcabcabd" を vocabSize 258 で学習した結果
	merges := []bpetrain.MergeRule{
		{Pair: bpetrain.Pair{First: 97, Second: 98}, NewID: 256},  // "ab"
		{Pair: bpetrain.Pair{First: 256, Second: 99}, NewID: 257}, // "abc"
	}
	tok, err := New(merges)
	if err != nil {
		t.Fatalf("New returned error: %v", err)
	}

	if got, want := tok.VocabSize(), bytetokenizer.VocabSize+len(merges); got != want {
		t.Errorf("VocabSize() = %d, want %d", got, want)
	}

	tests := []struct {
		id   int
		want []byte
	}{
		{97, []byte("a")},
		{256, []byte("ab")},
		{257, []byte("abc")},
	}
	for _, tt := range tests {
		got, ok := tok.Bytes(tt.id)
		if !ok {
			t.Errorf("Bytes(%d) = not found", tt.id)
			continue
		}
		if !reflect.DeepEqual(got, tt.want) {
			t.Errorf("Bytes(%d) = %q, want %q", tt.id, got, tt.want)
		}
	}

	if got := tok.Merges(); !reflect.DeepEqual(got, merges) {
		t.Errorf("Merges() = %v, want %v", got, merges)
	}
}

func TestNewMultibyte(t *testing.T) {
	// "ああ" (E3 81 82 E3 81 82) を vocabSize 258 で学習した結果
	merges := []bpetrain.MergeRule{
		{Pair: bpetrain.Pair{First: 0xE3, Second: 0x81}, NewID: 256},
		{Pair: bpetrain.Pair{First: 256, Second: 0x82}, NewID: 257}, // "あ"
	}
	tok, err := New(merges)
	if err != nil {
		t.Fatalf("New returned error: %v", err)
	}
	got, ok := tok.Bytes(257)
	if !ok {
		t.Fatal("Bytes(257) = not found")
	}
	if want := []byte("あ"); !reflect.DeepEqual(got, want) {
		t.Errorf("Bytes(257) = %v, want %v", got, want)
	}
}

func TestNewInvalidMerges(t *testing.T) {
	tests := []struct {
		name   string
		merges []bpetrain.MergeRule
	}{
		{
			"newID not starting at 256",
			[]bpetrain.MergeRule{{Pair: bpetrain.Pair{First: 97, Second: 98}, NewID: 300}},
		},
		{
			"newID not sequential",
			[]bpetrain.MergeRule{
				{Pair: bpetrain.Pair{First: 97, Second: 98}, NewID: 256},
				{Pair: bpetrain.Pair{First: 256, Second: 99}, NewID: 258},
			},
		},
		{
			"pair references future id",
			[]bpetrain.MergeRule{{Pair: bpetrain.Pair{First: 256, Second: 98}, NewID: 256}},
		},
		{
			"pair references unknown id",
			[]bpetrain.MergeRule{{Pair: bpetrain.Pair{First: 97, Second: 1000}, NewID: 256}},
		},
		{
			"pair references negative id",
			[]bpetrain.MergeRule{{Pair: bpetrain.Pair{First: -1, Second: 98}, NewID: 256}},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if _, err := New(tt.merges); err == nil {
				t.Errorf("New(%v) = nil error, want error", tt.merges)
			}
		})
	}
}

func TestBytesOutOfRange(t *testing.T) {
	tok, err := New(nil)
	if err != nil {
		t.Fatal(err)
	}
	for _, id := range []int{-1, 256, 1000} {
		if _, ok := tok.Bytes(id); ok {
			t.Errorf("Bytes(%d) = found, want not found", id)
		}
	}
}

// TestBytesReturnsCopy は Bytes の戻り値を書き換えても内部状態が変わらないことを確認する。
func TestBytesReturnsCopy(t *testing.T) {
	tok, err := New(nil)
	if err != nil {
		t.Fatal(err)
	}
	b, _ := tok.Bytes(97)
	b[0] = 0
	got, _ := tok.Bytes(97)
	if want := []byte{97}; !reflect.DeepEqual(got, want) {
		t.Errorf("internal vocab modified: Bytes(97) = %v, want %v", got, want)
	}
}

// TestNewDoesNotAliasInput は引数の merges を書き換えても内部状態が変わらないことを確認する。
func TestNewDoesNotAliasInput(t *testing.T) {
	merges := []bpetrain.MergeRule{{Pair: bpetrain.Pair{First: 97, Second: 98}, NewID: 256}}
	tok, err := New(merges)
	if err != nil {
		t.Fatal(err)
	}
	merges[0].NewID = 999
	if got := tok.Merges(); got[0].NewID != 256 {
		t.Errorf("internal merges modified: Merges()[0].NewID = %d, want 256", got[0].NewID)
	}
}

// TestNewFromTrain は bpetrain.Train の出力をそのまま受け取れ、
// 学習後の各トークンが元の文字列の部分バイト列に対応することを確認する。
func TestNewFromTrain(t *testing.T) {
	const text = "abcabcabcabdabdxyzxyz"
	ids, merges, err := bpetrain.Train(text, 266)
	if err != nil {
		t.Fatalf("Train returned error: %v", err)
	}
	tok, err := New(merges)
	if err != nil {
		t.Fatalf("New returned error: %v", err)
	}
	if got, want := tok.VocabSize(), bytetokenizer.VocabSize+len(merges); got != want {
		t.Errorf("VocabSize() = %d, want %d", got, want)
	}

	// 学習後の ID 列を語彙で展開すると元の文字列に戻る
	var buf []byte
	for _, id := range ids {
		b, ok := tok.Bytes(id)
		if !ok {
			t.Fatalf("Bytes(%d) = not found", id)
		}
		buf = append(buf, b...)
	}
	if got := string(buf); got != text {
		t.Errorf("expanded ids = %q, want %q", got, text)
	}
}
