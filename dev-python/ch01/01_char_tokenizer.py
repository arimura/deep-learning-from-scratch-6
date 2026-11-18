text = "hello世界😁"
print(list(text))

print(ord('h'))
print(ord('😁'))

print(chr(104))
print(chr(128513))

ids = [ord(char) for char in list(text)]
print(ids)

class CharTokenizer:
    def encode(self, text):
        return [ord(char) for char in text]

    def decode(self, ids):
        return ''.join([chr(i) for i in ids])

tokenizer = CharTokenizer()
text = "hello世界😁"

ids = tokenizer.encode(text)
print(ids)

decoded = tokenizer.decode(ids)
print(decoded)
