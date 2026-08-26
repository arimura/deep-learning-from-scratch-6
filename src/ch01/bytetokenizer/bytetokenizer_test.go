package bytetokenizer

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
		// "こ" = U+3053 → E3 81 93, "ん" = U+3093 → E3 82 93
		{"japanese", "こん", []int{0xE3, 0x81, 0x93, 0xE3, 0x82, 0x93}},
		// "😀" = U+1F600 → F0 9F 98 80
		{"emoji", "a😀", []int{97, 0xF0, 0x9F, 0x98, 0x80}},
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
		{"japanese", []int{0xE3, 0x81, 0x93, 0xE3, 0x82, 0x93}, "こん"},
		{"emoji", []int{97, 0xF0, 0x9F, 0x98, 0x80}, "a😀"},
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
	tests := []struct {
		name string
		in   []int
	}{
		{"negative", []int{-1}},
		{"out of range", []int{256}},
		{"truncated multibyte", []int{0xE3, 0x81}},
		{"lone continuation byte", []int{0x81}},
		{"invalid lead byte", []int{0xFF}},
		// UTF-8 でエンコードされたサロゲート (U+D800) は不正
		{"surrogate", []int{0xED, 0xA0, 0x80}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if _, err := tok.Decode(tt.in); err == nil {
				t.Errorf("Decode(%v) expected error, got nil", tt.in)
			}
		})
	}
}

func TestRoundTrip(t *testing.T) {
	tok := New()
	in := "Hello, 世界! 🌏\n"
	ids, err := tok.Encode(in)
	if err != nil {
		t.Fatal(err)
	}
	for i, id := range ids {
		if id < 0 || id >= VocabSize {
			t.Fatalf("id %d at index %d out of range [0, %d)", id, i, VocabSize)
		}
	}
	out, err := tok.Decode(ids)
	if err != nil {
		t.Fatal(err)
	}
	if out != in {
		t.Errorf("round trip = %q, want %q", out, in)
	}
}
