// Command chartokenizer は文字列を Unicode コードポイント列に encode / decode する CLI。
//
// 使い方:
//
//	go run ./cmd/chartokenizer encode "こんにちは"
//	go run ./cmd/chartokenizer decode 12371 12435 12395 12385 12399
//	echo "hello" | go run ./cmd/chartokenizer encode
//	echo "104 101 108 108 111" | go run ./cmd/chartokenizer decode
package main

import (
	"bufio"
	"fmt"
	"io"
	"os"
	"strconv"
	"strings"

	"github.com/arimura/deep-learning-from-scratch-6/dev/ch01/chartokenizer"
)

func usage() {
	fmt.Fprintln(os.Stderr, "usage:")
	fmt.Fprintln(os.Stderr, "  chartokenizer encode [text]        # 文字列 → コードポイント列 (省略時は stdin)")
	fmt.Fprintln(os.Stderr, "  chartokenizer decode [id ...]      # コードポイント列 → 文字列 (省略時は stdin)")
}

func main() {
	if len(os.Args) < 2 {
		usage()
		os.Exit(2)
	}

	tok := chartokenizer.New()
	var err error
	switch os.Args[1] {
	case "encode":
		err = runEncode(tok, os.Args[2:])
	case "decode":
		err = runDecode(tok, os.Args[2:])
	default:
		usage()
		os.Exit(2)
	}
	if err != nil {
		fmt.Fprintln(os.Stderr, "error:", err)
		os.Exit(1)
	}
}

func runEncode(tok *chartokenizer.Tokenizer, args []string) error {
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

func runDecode(tok *chartokenizer.Tokenizer, args []string) error {
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
		ids = append(ids, id)
	}

	s, err := tok.Decode(ids)
	if err != nil {
		return err
	}
	fmt.Println(s)
	return nil
}
