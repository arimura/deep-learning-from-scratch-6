// Command bpetrain は文字列に対して BPE の学習を行い、
// 得られたマージ規則と学習後のトークン ID 列を表示する CLI。
//
// 使い方:
//
//	go run ./cmd/bpetrain -vocab 300 "abcabcabd"
//	cat corpus.txt | go run ./cmd/bpetrain -vocab 512
package main

import (
	"bufio"
	"flag"
	"fmt"
	"io"
	"os"
	"strconv"
	"strings"

	"github.com/arimura/deep-learning-from-scratch-6/dev/ch01/bpetrain"
	"github.com/arimura/deep-learning-from-scratch-6/dev/ch01/bytetokenizer"
)

func main() {
	vocabSize := flag.Int("vocab", 276, "学習後のボキャブラリーサイズ (257 以上、特殊トークン用の 1 語を含む)")
	special := flag.String("special", bpetrain.DefaultSpecialToken, "学習対象の文字列を分割する特殊トークン")
	showIDs := flag.Bool("ids", false, "学習後のトークン ID 列も表示する")
	flag.Usage = func() {
		fmt.Fprintln(os.Stderr, "usage: bpetrain [-vocab N] [-special TOKEN] [-ids] [text]")
		fmt.Fprintln(os.Stderr, "  text を省略した場合は stdin から読み込む")
		flag.PrintDefaults()
	}
	flag.Parse()

	if err := run(*vocabSize, *special, *showIDs, flag.Args()); err != nil {
		fmt.Fprintln(os.Stderr, "error:", err)
		os.Exit(1)
	}
}

func run(vocabSize int, special string, showIDs bool, args []string) error {
	text, err := readText(args)
	if err != nil {
		return err
	}
	if text == "" {
		return fmt.Errorf("学習対象の文字列が空です")
	}

	chunks, merges, err := bpetrain.TrainWithSpecialToken(text, vocabSize, special)
	if err != nil {
		return err
	}

	// 入力トークン数は特殊トークンを除いたバイト数で数える
	inputTokens := 0
	for _, part := range strings.Split(text, special) {
		inputTokens += len(part)
	}
	outputTokens := 0
	for _, ids := range chunks {
		outputTokens += len(ids)
	}

	fmt.Printf("分割数: %d\n", len(chunks))
	fmt.Printf("入力トークン数: %d\n", inputTokens)
	fmt.Printf("学習後トークン数: %d\n", outputTokens)
	if outputTokens > 0 {
		fmt.Printf("圧縮率: %.2fX\n", float64(inputTokens)/float64(outputTokens))
	}
	fmt.Printf("マージ回数: %d\n", len(merges))

	fmt.Println("マージ規則:")
	vocab := buildVocab(merges)
	for _, m := range merges {
		fmt.Printf("  %d <- (%d, %d)  %q\n",
			m.NewID, m.Pair.First, m.Pair.Second, string(vocab[m.NewID]))
	}

	if showIDs {
		fmt.Println("トークン ID 列 (特殊トークンごとに分割):")
		for _, ids := range chunks {
			strs := make([]string, len(ids))
			for i, id := range ids {
				strs[i] = strconv.Itoa(id)
			}
			fmt.Println(" ", strings.Join(strs, " "))
		}
	}
	return nil
}

func readText(args []string) (string, error) {
	if len(args) > 0 {
		return strings.Join(args, " "), nil
	}
	b, err := io.ReadAll(bufio.NewReader(os.Stdin))
	if err != nil {
		return "", err
	}
	return strings.TrimRight(string(b), "\n"), nil
}

// buildVocab は各トークン ID が表すバイト列を組み立てる。
// 0〜255 は対応するバイトそのもの、256 以降はマージ元 2 つの連結である。
func buildVocab(merges []bpetrain.MergeRule) map[int][]byte {
	vocab := make(map[int][]byte, bytetokenizer.VocabSize+len(merges))
	for i := 0; i < bytetokenizer.VocabSize; i++ {
		vocab[i] = []byte{byte(i)}
	}
	for _, m := range merges {
		b := make([]byte, 0, len(vocab[m.Pair.First])+len(vocab[m.Pair.Second]))
		b = append(b, vocab[m.Pair.First]...)
		b = append(b, vocab[m.Pair.Second]...)
		vocab[m.NewID] = b
	}
	return vocab
}
