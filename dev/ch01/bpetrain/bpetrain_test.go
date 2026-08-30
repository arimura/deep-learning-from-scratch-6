package bpetrain

import (
	"reflect"
	"testing"
)

func TestCountPairs(t *testing.T) {
	tests := []struct {
		name string
		in   []int
		want map[Pair]int
	}{
		{"empty", []int{}, map[Pair]int{}},
		{"single", []int{1}, map[Pair]int{}},
		{
			"two",
			[]int{1, 2},
			map[Pair]int{{1, 2}: 1},
		},
		{
			"repeated pair",
			[]int{1, 2, 1, 2},
			map[Pair]int{{1, 2}: 2, {2, 1}: 1},
		},
		{
			// 隣接位置をすべて独立に数えるため {1,1} は 2 回
			"same id run",
			[]int{1, 1, 1},
			map[Pair]int{{1, 1}: 2},
		},
		{
			// "abcabcabd" をバイト値にしたもの
			"abcabcabd",
			[]int{97, 98, 99, 97, 98, 99, 97, 98, 100},
			map[Pair]int{
				{97, 98}:  3,
				{98, 99}:  2,
				{99, 97}:  2,
				{98, 100}: 1,
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := CountPairs(tt.in, nil)
			if !reflect.DeepEqual(got, tt.want) {
				t.Errorf("CountPairs(%v, nil) = %v, want %v", tt.in, got, tt.want)
			}
		})
	}
}

func TestCountPairsDoesNotModifyInput(t *testing.T) {
	in := []int{1, 2, 3}
	want := []int{1, 2, 3}
	CountPairs(in, nil)
	if !reflect.DeepEqual(in, want) {
		t.Errorf("CountPairs modified input: got %v, want %v", in, want)
	}
}

// TestCountPairsAccumulates は counts を渡した場合に既存の値へ加算されることを確認する。
func TestCountPairsAccumulates(t *testing.T) {
	counts := map[Pair]int{{1, 2}: 5, {9, 9}: 1}
	got := CountPairs([]int{1, 2, 3, 1, 2}, counts)

	want := map[Pair]int{{1, 2}: 7, {2, 3}: 1, {3, 1}: 1, {9, 9}: 1}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("CountPairs with counts = %v, want %v", got, want)
	}
	// 渡した map そのものが更新され、同じ map が返る
	if !reflect.DeepEqual(counts, want) {
		t.Errorf("passed counts = %v, want %v", counts, want)
	}
	if reflect.ValueOf(got).Pointer() != reflect.ValueOf(counts).Pointer() {
		t.Error("CountPairs returned a different map than the one passed in")
	}
}

// TestCountPairsAcrossChunks は複数のトークン ID 列を順に渡して集約できることを確認する。
func TestCountPairsAcrossChunks(t *testing.T) {
	chunks := [][]int{
		{97, 98, 99},
		{97, 98, 100},
		{},
		{98},
	}
	var counts map[Pair]int
	for _, c := range chunks {
		counts = CountPairs(c, counts)
	}
	want := map[Pair]int{{97, 98}: 2, {98, 99}: 1, {98, 100}: 1}
	if !reflect.DeepEqual(counts, want) {
		t.Errorf("accumulated counts = %v, want %v", counts, want)
	}
}

// TestCountPairsEmptyWithCounts は空の ids でも渡した counts がそのまま返ることを確認する。
func TestCountPairsEmptyWithCounts(t *testing.T) {
	counts := map[Pair]int{{1, 2}: 3}
	got := CountPairs(nil, counts)
	want := map[Pair]int{{1, 2}: 3}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("CountPairs(nil, counts) = %v, want %v", got, want)
	}
}

func TestMerge(t *testing.T) {
	tests := []struct {
		name  string
		in    []int
		pair  Pair
		newID int
		want  []int
	}{
		{"empty", []int{}, Pair{1, 2}, 256, []int{}},
		{"single", []int{1}, Pair{1, 2}, 256, []int{1}},
		{"no match", []int{1, 3, 2}, Pair{1, 2}, 256, []int{1, 3, 2}},
		{"head", []int{1, 2, 3}, Pair{1, 2}, 256, []int{256, 3}},
		{"tail", []int{3, 1, 2}, Pair{1, 2}, 256, []int{3, 256}},
		{"whole", []int{1, 2}, Pair{1, 2}, 256, []int{256}},
		{
			"multiple",
			[]int{1, 2, 3, 1, 2},
			Pair{1, 2}, 256,
			[]int{256, 3, 256},
		},
		{
			// 一度置き換えた要素は次のペアの前方要素にしない
			"overlapping run",
			[]int{1, 1, 1},
			Pair{1, 1}, 256,
			[]int{256, 1},
		},
		{
			"overlapping run even",
			[]int{1, 1, 1, 1},
			Pair{1, 1}, 256,
			[]int{256, 256},
		},
		{
			"asymmetric pair",
			[]int{2, 1, 2, 1},
			Pair{1, 2}, 256,
			[]int{2, 256, 1},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := Merge(tt.in, tt.pair, tt.newID)
			if !reflect.DeepEqual(got, tt.want) {
				t.Errorf("Merge(%v, %v, %d) = %v, want %v", tt.in, tt.pair, tt.newID, got, tt.want)
			}
		})
	}
}

func TestMergeDoesNotModifyInput(t *testing.T) {
	in := []int{1, 2, 3, 1, 2}
	want := []int{1, 2, 3, 1, 2}
	Merge(in, Pair{1, 2}, 256)
	if !reflect.DeepEqual(in, want) {
		t.Errorf("Merge modified input: got %v, want %v", in, want)
	}
}

// TestTrainStep は CountPairs と Merge を組み合わせた BPE 学習 1 ステップの動作を確認する。
func TestTrainStep(t *testing.T) {
	// "abcabcabd" のバイト値
	ids := []int{97, 98, 99, 97, 98, 99, 97, 98, 100}

	counts := CountPairs(ids, nil)
	var best Pair
	bestCount := 0
	for p, c := range counts {
		if c > bestCount {
			best, bestCount = p, c
		}
	}
	if want := (Pair{97, 98}); best != want || bestCount != 3 {
		t.Fatalf("most frequent pair = %v (%d times), want %v (3 times)", best, bestCount, want)
	}

	got := Merge(ids, best, 256)
	want := []int{256, 99, 256, 99, 256, 100}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("Merge(%v, %v, 256) = %v, want %v", ids, best, got, want)
	}
}

// TestTrain は Train の動作を確認する。
// ボキャブラリーサイズは初期語彙 256 に特殊トークン用の 1 語を加えた 257 が
// マージ回数 0 に対応する。
func TestTrain(t *testing.T) {
	tests := []struct {
		name       string
		text       string
		vocabSize  int
		wantIDs    [][]int
		wantMerges []MergeRule
	}{
		{
			// マージ回数 0 なのでバイト列のまま
			"no merge",
			"abc", 257,
			[][]int{{97, 98, 99}},
			[]MergeRule{},
		},
		{
			// {97,98} が 3 回で最頻 → 256
			"single merge",
			"abcabcabd", 258,
			[][]int{{256, 99, 256, 99, 256, 100}},
			[]MergeRule{{Pair{97, 98}, 256}},
		},
		{
			// 続けて {256,99} が 2 回で最頻 → 257
			"two merges",
			"abcabcabd", 259,
			[][]int{{257, 257, 256, 100}},
			[]MergeRule{{Pair{97, 98}, 256}, {Pair{256, 99}, 257}},
		},
		{
			// マージできるペアが無くなればマージ回数に達しなくても打ち切る
			"exhausted",
			"aaaa", 300,
			[][]int{{257}},
			[]MergeRule{{Pair{97, 97}, 256}, {Pair{256, 256}, 257}},
		},
		{
			"single byte input",
			"a", 300,
			[][]int{{97}},
			[]MergeRule{},
		},
		{
			// マルチバイト文字も UTF-8 バイト列として扱う
			"multibyte",
			"ああ", 258,
			[][]int{{256, 130, 256, 130}},
			[]MergeRule{{Pair{227, 129}, 256}},
		},
		{
			// 特殊トークンで分割され、出現回数は全配列にわたって集約される
			"split by special token",
			"ab<|endoftext|>ab", 258,
			[][]int{{256}, {256}},
			[]MergeRule{{Pair{97, 98}, 256}},
		},
		{
			// 特殊トークンをまたぐペア {98,97} はカウントされない
			"no pair across special token",
			"ab<|endoftext|>ab<|endoftext|>ab", 300,
			[][]int{{256}, {256}, {256}},
			[]MergeRule{{Pair{97, 98}, 256}},
		},
		{
			// 先頭・末尾・連続する特殊トークンは空の配列になる
			"empty chunks",
			"<|endoftext|>ab<|endoftext|><|endoftext|>", 258,
			[][]int{{}, {256}, {}, {}},
			[]MergeRule{{Pair{97, 98}, 256}},
		},
		{
			// 特殊トークンのみの文字列でもエラーにならない
			"only special token",
			"<|endoftext|>", 300,
			[][]int{{}, {}},
			[]MergeRule{},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			ids, merges, err := Train(tt.text, tt.vocabSize)
			if err != nil {
				t.Fatalf("Train(%q, %d) returned error: %v", tt.text, tt.vocabSize, err)
			}
			if !reflect.DeepEqual(ids, tt.wantIDs) {
				t.Errorf("ids = %v, want %v", ids, tt.wantIDs)
			}
			if !reflect.DeepEqual(merges, tt.wantMerges) {
				t.Errorf("merges = %v, want %v", merges, tt.wantMerges)
			}
		})
	}
}

// TestTrainWithSpecialToken は特殊トークンを差し替えられることを確認する。
func TestTrainWithSpecialToken(t *testing.T) {
	ids, merges, err := TrainWithSpecialToken("ab|ba", 258, "|")
	if err != nil {
		t.Fatalf("TrainWithSpecialToken returned error: %v", err)
	}
	// {97,98} と {98,97} が 1 回ずつで同数なので、先に現れる {97,98} を選ぶ
	wantIDs := [][]int{{256}, {98, 97}}
	wantMerges := []MergeRule{{Pair{97, 98}, 256}}
	if !reflect.DeepEqual(ids, wantIDs) {
		t.Errorf("ids = %v, want %v", ids, wantIDs)
	}
	if !reflect.DeepEqual(merges, wantMerges) {
		t.Errorf("merges = %v, want %v", merges, wantMerges)
	}

	// 既定の特殊トークンは分割に使われない
	ids, _, err = TrainWithSpecialToken("a<|endoftext|>a", 300, "|")
	if err != nil {
		t.Fatalf("TrainWithSpecialToken returned error: %v", err)
	}
	if len(ids) != 3 {
		t.Errorf("len(ids) = %d, want 3 (split by %q, not by default token)", len(ids), "|")
	}
}

func TestTrainEmptySpecialToken(t *testing.T) {
	if _, _, err := TrainWithSpecialToken("abc", 300, ""); err == nil {
		t.Error("TrainWithSpecialToken with empty special token = nil error, want error")
	}
}

// TestTrainVocabSizeTooSmall は特殊トークン用の 1 語を含めた最小語彙サイズ (257)
// 未満を指定するとエラーになることを確認する。
func TestTrainVocabSizeTooSmall(t *testing.T) {
	for _, vocabSize := range []int{0, 255, 256} {
		if _, _, err := Train("abc", vocabSize); err == nil {
			t.Errorf("Train with vocabSize %d = nil error, want error", vocabSize)
		}
	}
}

// TestTrainNumMerges はマージ回数が vocabSize - 257 になることを確認する。
func TestTrainNumMerges(t *testing.T) {
	// ペアが枯渇しないよう十分に多様な文字列を使う
	const text = "abcabcabcabdabdxyzxyz"
	for _, vocabSize := range []int{257, 258, 260} {
		_, merges, err := Train(text, vocabSize)
		if err != nil {
			t.Fatalf("Train returned error: %v", err)
		}
		if want := vocabSize - MinVocabSize; len(merges) != want {
			t.Errorf("Train(%q, %d): len(merges) = %d, want %d", text, vocabSize, len(merges), want)
		}
	}
}

func TestTrainInvalidUTF8(t *testing.T) {
	if _, _, err := Train(string([]byte{0xff, 0xfe}), 300); err == nil {
		t.Error("Train with invalid UTF-8 = nil error, want error")
	}
}

// TestTrainDeterministic は同数のペアが複数ある場合でも結果が一定であることを確認する。
func TestTrainDeterministic(t *testing.T) {
	// {97,98} {98,97} {97,99} {99,97} {97,100} がいずれも 1 回ずつ
	const text = "abacad"
	ids, merges, err := Train(text, 258)
	if err != nil {
		t.Fatalf("Train returned error: %v", err)
	}
	for i := 0; i < 20; i++ {
		gotIDs, gotMerges, err := Train(text, 258)
		if err != nil {
			t.Fatalf("Train returned error: %v", err)
		}
		if !reflect.DeepEqual(gotIDs, ids) || !reflect.DeepEqual(gotMerges, merges) {
			t.Fatalf("Train is not deterministic: (%v, %v) != (%v, %v)", gotIDs, gotMerges, ids, merges)
		}
	}
}

// TestTrainNewIDsAreSequential は新しいトークン ID が 256 から連番であることを確認する。
func TestTrainNewIDsAreSequential(t *testing.T) {
	_, merges, err := Train("abcabcabcabdabdxyzxyz", 266)
	if err != nil {
		t.Fatalf("Train returned error: %v", err)
	}
	for i, m := range merges {
		if want := 256 + i; m.NewID != want {
			t.Errorf("merges[%d].NewID = %d, want %d", i, m.NewID, want)
		}
	}
}
