from collections import defaultdict

def count_pairs(ids):
    counts = defaultdict(int)
    for pair in zip(ids, ids[1:]):
        counts[pair] += 1
    return counts

ids = [1, 2, 3, 1, 2]
counts = count_pairs(ids)
print(counts)

def merge(ids, pair, new_id):
    merged_ids = []
    i = 0

    while i < len(ids):
        if i < len(ids) - 1 and (ids[i], ids[i+1]) == pair:
            merged_ids.append(new_id)
            i += 2
        else:
            merged_ids.append(ids[i])
            i += 1
    return merged_ids

ids = [1, 2, 3, 1, 2]
merged = merge(ids, (1,2), 4)
print(merged)

def train_bpe(text, vocab_size):
    ids = list(text.encode("utf-8"))

    num_merges = vocab_size - 256
    merge_rules = {}

    for step in range(num_merges):
        counts = count_pairs(ids)

        if not counts:
            break

        best_pair = max(counts, key=counts.get)

        new_id = 256 + step
        merge_rules[best_pair] = new_id

        ids = merge(ids, best_pair, new_id)

    return merge_rules

text = "Hello world! This is BPE training."
merge_rules = train_bpe(text, vocab_size=260)
print(merge_rules)
