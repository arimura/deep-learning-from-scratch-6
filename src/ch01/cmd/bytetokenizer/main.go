// Command bytetokenizer は文字列を UTF-8 バイト列に encode / decode する CLI。
//
// 使い方:
//
//	go run ./cmd/bytetokenizer encode "こんにちは"
//	go run ./cmd/bytetokenizer decode 227 129 147 227 130 147 227 129 171 227 129 161 227 129 175
//	echo "hello" | go run ./cmd/bytetokenizer encode
//	echo "104 101 108 108 111" | go run ./cmd/bytetokenizer decode
package main

import (
	"bufio"
	"fmt"
	"io"
	"os"
	"strconv"
	"strings"

	"ch01/bytetokenizer"
)

func usage() {
	fmt.Fprintln(os.Stderr, "usage:")
	fmt.Fprintln(os.Stderr, "  bytetokenizer encode [text]        # 文字列 → バイト値列 (省略時は stdin)")
	fmt.Fprintln(os.Stderr, "  bytetokenizer decode [id ...]      # バイト値列 → 文字列 (省略時は stdin)")
}

func main() {
	if len(os.Args) < 2 {
		usage()
		os.Exit(2)
	}

	tok := bytetokenizer.New()
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

func runEncode(tok *bytetokenizer.Tokenizer, args []string) error {
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

func runDecode(tok *bytetokenizer.Tokenizer, args []string) error {
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
