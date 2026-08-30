package bpetokenizer

import (
	"reflect"
	"testing"

	"github.com/arimura/deep-learning-from-scratch-6/dev/ch01/bpetrain"
	"github.com/arimura/deep-learning-from-scratch-6/dev/ch01/bytetokenizer"
)

// flatten は bpetrain.Train が返すトークン ID 列の配列を 1 つの列に連結する。
func flatten(chunks [][]int) []int {
	var ids []int
	for _, c := range chunks {
		ids = append(ids, c...)
	}
	return ids
}

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
	// (text に特殊トークンは含まれないので分割は 1 つ)
	var buf []byte
	for _, id := range flatten(ids) {
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

func TestEncode(t *testing.T) {
	// "abcabcabd" を vocabSize 258 で学習した結果
	merges := []bpetrain.MergeRule{
		{Pair: bpetrain.Pair{First: 97, Second: 98}, NewID: 256},  // "ab"
		{Pair: bpetrain.Pair{First: 256, Second: 99}, NewID: 257}, // "abc"
	}
	tok, err := New(merges)
	if err != nil {
		t.Fatalf("New returned error: %v", err)
	}

	tests := []struct {
		name string
		in   string
		want []int
	}{
		{"empty", "", []int{}},
		{"single byte", "a", []int{97}},
		{"no merge applies", "xyz", []int{120, 121, 122}},
		{"first merge only", "abd", []int{256, 100}},
		{"chained merges", "abcabcabd", []int{257, 257, 256, 100}},
		{"merge order matters", "abcbc", []int{257, 98, 99}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := tok.Encode(tt.in)
			if err != nil {
				t.Fatalf("Encode(%q) returned error: %v", tt.in, err)
			}
			if !reflect.DeepEqual(got, tt.want) {
				t.Errorf("Encode(%q) = %v, want %v", tt.in, got, tt.want)
			}
		})
	}
}

func TestEncodeNoMerges(t *testing.T) {
	// マージ規則が無い場合は bytetokenizer と同じ結果になる
	tok, err := New(nil)
	if err != nil {
		t.Fatal(err)
	}
	const text = "aあ😀"
	got, err := tok.Encode(text)
	if err != nil {
		t.Fatalf("Encode returned error: %v", err)
	}
	want, _ := bytetokenizer.New().Encode(text)
	if !reflect.DeepEqual(got, want) {
		t.Errorf("Encode(%q) = %v, want %v", text, got, want)
	}
}

func TestEncodeInvalidUTF8(t *testing.T) {
	tok, err := New(nil)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := tok.Encode("a\xffb"); err == nil {
		t.Error("Encode(invalid UTF-8) = nil error, want error")
	}
}

func TestDecode(t *testing.T) {
	// "ああ" (E3 81 82 E3 81 82) を vocabSize 258 で学習した結果
	merges := []bpetrain.MergeRule{
		{Pair: bpetrain.Pair{First: 0xE3, Second: 0x81}, NewID: 256},
		{Pair: bpetrain.Pair{First: 256, Second: 0x82}, NewID: 257}, // "あ"
	}
	tok, err := New(merges)
	if err != nil {
		t.Fatalf("New returned error: %v", err)
	}

	tests := []struct {
		name string
		in   []int
		want string
	}{
		{"empty", []int{}, ""},
		{"nil", nil, ""},
		{"bytes only", []int{97, 98, 99}, "abc"},
		{"merged token", []int{257}, "あ"},
		{"partial merge plus byte", []int{256, 0x82}, "あ"},
		{"mixed", []int{97, 257, 257, 98}, "aああb"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := tok.Decode(tt.in)
			if err != nil {
				t.Fatalf("Decode(%v) returned error: %v", tt.in, err)
			}
			if got != tt.want {
				t.Errorf("Decode(%v) = %q, want %q", tt.in, got, tt.want)
			}
		})
	}
}

func TestDecodeInvalid(t *testing.T) {
	merges := []bpetrain.MergeRule{
		{Pair: bpetrain.Pair{First: 0xE3, Second: 0x81}, NewID: 256},
	}
	tok, err := New(merges)
	if err != nil {
		t.Fatal(err)
	}
	tests := []struct {
		name string
		in   []int
	}{
		{"negative id", []int{-1}},
		{"id beyond vocab", []int{257}},
		{"far out of range", []int{97, 1000}},
		{"incomplete utf-8", []int{256}}, // E3 81 だけでは不完全
		{"invalid utf-8 byte", []int{0xFF}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if _, err := tok.Decode(tt.in); err == nil {
				t.Errorf("Decode(%v) = nil error, want error", tt.in)
			}
		})
	}
}

// TestRoundTrip は Train で得たマージ規則を用いて Encode → Decode すると
// 元の文字列に戻ることを確認する。学習に使った文字列以外も対象とする。
func TestRoundTrip(t *testing.T) {
	const corpus = "こんにちは世界。こんにちは BPE。hello world hello bpe"
	trainIDs, merges, err := bpetrain.Train(corpus, 300)
	if err != nil {
		t.Fatalf("Train returned error: %v", err)
	}
	tok, err := New(merges)
	if err != nil {
		t.Fatalf("New returned error: %v", err)
	}

	// 学習対象の文字列は Train の出力と同じ ID 列にエンコードされる
	// (corpus に特殊トークンは含まれないので分割は 1 つ)
	got, err := tok.Encode(corpus)
	if err != nil {
		t.Fatalf("Encode returned error: %v", err)
	}
	if want := flatten(trainIDs); !reflect.DeepEqual(got, want) {
		t.Errorf("Encode(corpus) = %v, want Train ids %v", got, want)
	}

	texts := []string{
		"",
		corpus,
		"hello",
		"こんにちは",
		"未知の文字列 with emoji 😀 and\nnewline",
	}
	for _, text := range texts {
		ids, err := tok.Encode(text)
		if err != nil {
			t.Errorf("Encode(%q) returned error: %v", text, err)
			continue
		}
		for _, id := range ids {
			if id < 0 || id >= tok.VocabSize() {
				t.Errorf("Encode(%q) produced out-of-vocab id %d", text, id)
			}
		}
		back, err := tok.Decode(ids)
		if err != nil {
			t.Errorf("Decode(Encode(%q)) returned error: %v", text, err)
			continue
		}
		if back != text {
			t.Errorf("Decode(Encode(%q)) = %q", text, back)
		}
	}
}
