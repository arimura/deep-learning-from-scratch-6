// Command bpetokenizer は BPE トークナイザで文字列を encode / decode する CLI。
//
// マージ規則はこのファイルに直接書かれたもの (merges) を用いる。
//
// 使い方:
//
//	go run ./cmd/bpetokenizer encode "hello こんにちは"
//	go run ./cmd/bpetokenizer decode 260 271          # "hello こんにちは"
//	echo "hello" | go run ./cmd/bpetokenizer encode
//	echo "260" | go run ./cmd/bpetokenizer decode
//	go run ./cmd/bpetokenizer vocab        # マージ規則の一覧を表示
package main

import (
	"bufio"
	"fmt"
	"io"
	"os"
	"strconv"
	"strings"

	"github.com/arimura/deep-learning-from-scratch-6/dev/ch01/bpetokenizer"
	"github.com/arimura/deep-learning-from-scratch-6/dev/ch01/bpetrain"
)

// merges は CLI が使うマージ規則。
var merges = []bpetrain.MergeRule{
	{Pair: bpetrain.Pair{First: 105, Second: 115}, NewID: 256},
	{Pair: bpetrain.Pair{First: 256, Second: 32}, NewID: 257},
	{Pair: bpetrain.Pair{First: 105, Second: 110}, NewID: 258},
	{Pair: bpetrain.Pair{First: 72, Second: 101}, NewID: 259},
}

func usage() {
	fmt.Fprintln(os.Stderr, "usage:")
	fmt.Fprintln(os.Stderr, "  bpetokenizer encode [text]      # 文字列 → トークン ID 列 (省略時は stdin)")
	fmt.Fprintln(os.Stderr, "  bpetokenizer decode [id ...]    # トークン ID 列 → 文字列 (省略時は stdin)")
	fmt.Fprintln(os.Stderr, "  bpetokenizer vocab              # マージ規則の一覧を表示")
}

func main() {
	if len(os.Args) < 2 {
		usage()
		os.Exit(2)
	}

	tok, err := bpetokenizer.New(merges)
	if err != nil {
		fmt.Fprintln(os.Stderr, "error:", err)
		os.Exit(1)
	}

	switch os.Args[1] {
	case "encode":
		err = runEncode(tok, os.Args[2:])
	case "decode":
		err = runDecode(tok, os.Args[2:])
	case "vocab":
		runVocab(tok)
	default:
		usage()
		os.Exit(2)
	}
	if err != nil {
		fmt.Fprintln(os.Stderr, "error:", err)
		os.Exit(1)
	}
}

func runEncode(tok *bpetokenizer.Tokenizer, args []string) error {
	var text string
	if len(args) > 0 {
		text = strings.Join(args, " ")
	} else {
		b, err := io.ReadAll(bufio.NewReader(os.Stdin))
		if err != nil {
			return err
		}
		text = strings.TrimRight(string(b), "\n")
	}

	ids, err := tok.Encode(text)
	if err != nil {
		return err
	}
	strs := make([]string, len(ids))
	for i, id := range ids {
		strs[i] = strconv.Itoa(id)
	}
	fmt.Println(strings.Join(strs, " "))
	return nil
}

func runDecode(tok *bpetokenizer.Tokenizer, args []string) error {
	fields := args
	if len(fields) == 0 {
		b, err := io.ReadAll(bufio.NewReader(os.Stdin))
		if err != nil {
			return err
		}
		fields = strings.Fields(string(b))
	}

	ids := make([]int, 0, len(fields))
	for _, f := range fields {
		id, err := strconv.Atoi(f)
		if err != nil {
			return fmt.Errorf("invalid id %q: %w", f, err)
		}
		if id < 0 || id >= tok.VocabSize() {
			return fmt.Errorf("token id %d is out of range [0, %d)", id, tok.VocabSize())
		}
		ids = append(ids, id)
	}

	s, err := tok.Decode(ids)
	if err != nil {
		return err
	}
	fmt.Println(s)
	return nil
}

func runVocab(tok *bpetokenizer.Tokenizer) {
	fmt.Printf("語彙サイズ: %d (マージ回数: %d)\n", tok.VocabSize(), len(merges))
	for _, m := range tok.Merges() {
		b, _ := tok.Bytes(m.NewID)
		fmt.Printf("  %d <- (%d, %d)  %q\n", m.NewID, m.Pair.First, m.Pair.Second, string(b))
	}
}
