import re
import pickle

def merge_pair(ids, pair, new_id):
    a, b = pair
    new_word = []
    i = 0
    while i < len(ids):
        if i < len(ids) - 1 and ids[i] == a and ids[i + 1] == b:
            new_word.append(new_id)
            i += 2
        else:
            new_word.append(ids[i])
            i += 1
    return new_word

class BPETokenizer:
    def __init__(self, vocab, merges):
        self.vocab = vocab
        self.merges = merges
        self.exp = re.compile(r" ?[a-zA-Z]+| ?[0-9]+| ?[^\w\s]+|\s+")

    def _encode_chunk(self, chunk_str):
        word_ids = [b + 4 for b in chunk_str.encode("utf-8")]
        while len(word_ids) >= 2:
            pairs = [(word_ids[i], word_ids[i + 1]) for i in range(len(word_ids) - 1)]
            
            best_pair = min(pairs, key=lambda p: self.merges.get(p, float("inf")))
            if best_pair not in self.merges:
                break
            new_id = self.merges[best_pair]
            word_ids = merge_pair(word_ids, best_pair, new_id)
        return word_ids

    def encode(self, text):
        token_ids = []
        for chunk in self.exp.findall(text):
            token_ids.extend(self._encode_chunk(chunk))
        return token_ids

    def decode(self, token_ids):
        byte_chunks = []
        for tid in token_ids:
            byte_chunks.append(self.vocab[tid])
        return b"".join(byte_chunks).decode("utf-8", errors="replace")

def load_bpe_tokenizer():
    with open("bpe_checkpoint.pkl", "rb") as f:
        ckpt = pickle.load(f)
    merges = ckpt["merges"]
    vocab = ckpt["vocab"]
    return BPETokenizer(vocab, merges)