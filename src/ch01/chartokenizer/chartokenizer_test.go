package chartokenizer

import (
	"reflect"
	"testing"
)

func TestEncode(t *testing.T) {
	tok := New()
	tests := []struct {
		name string
		in   string
		want []int
	}{
		{"empty", "", []int{}},
		{"ascii", "abc", []int{97, 98, 99}},
		{"japanese", "こんにちは", []int{0x3053, 0x3093, 0x306B, 0x3061, 0x306F}},
		{"emoji", "a😀", []int{97, 0x1F600}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := tok.Encode(tt.in)
			if err != nil {
				t.Fatalf("Encode(%q) error = %v", tt.in, err)
			}
			if !reflect.DeepEqual(got, tt.want) {
				t.Errorf("Encode(%q) = %v, want %v", tt.in, got, tt.want)
			}
		})
	}
}

func TestEncodeInvalidUTF8(t *testing.T) {
	tok := New()
	if _, err := tok.Encode("\xff\xfe"); err == nil {
		t.Errorf("Encode(invalid utf8) expected error, got nil")
	}
}

func TestDecode(t *testing.T) {
	tok := New()
	tests := []struct {
		name string
		in   []int
		want string
	}{
		{"empty", []int{}, ""},
		{"ascii", []int{97, 98, 99}, "abc"},
		{"japanese", []int{0x3053, 0x3093, 0x306B, 0x3061, 0x306F}, "こんにちは"},
		{"emoji", []int{97, 0x1F600}, "a😀"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := tok.Decode(tt.in)
			if err != nil {
				t.Fatalf("Decode(%v) error = %v", tt.in, err)
			}
			if got != tt.want {
				t.Errorf("Decode(%v) = %q, want %q", tt.in, got, tt.want)
			}
		})
	}
}

func TestDecodeInvalid(t *testing.T) {
	tok := New()
	for _, ids := range [][]int{{-1}, {0xD800}, {0x110000}} {
		if _, err := tok.Decode(ids); err == nil {
			t.Errorf("Decode(%v) expected error, got nil", ids)
		}
	}
}

func TestRoundTrip(t *testing.T) {
	tok := New()
	in := "Hello, 世界! 🌏\n"
	ids, err := tok.Encode(in)
	if err != nil {
		t.Fatal(err)
	}
	out, err := tok.Decode(ids)
	if err != nil {
		t.Fatal(err)
	}
	if out != in {
		t.Errorf("round trip = %q, want %q", out, in)
	}
}
