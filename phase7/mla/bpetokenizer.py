import re
import pickle

def merge_pair_in_word(word_ids, pair_to_merge, new_id):
    """
    Scans a single word (list of IDs) and replaces every occurrence of (a, b) with new_id.
    """
    a, b = pair_to_merge
    new_word = []
    i = 0
    while i < len(word_ids):
        # If current item and next item match our pair, replace both with new_id
        if i < len(word_ids) - 1 and word_ids[i] == a and word_ids[i + 1] == b:
            new_word.append(new_id)
            i += 2  # Skip both elements since we merged them
        else:
            new_word.append(word_ids[i])
            i += 1
    return new_word

class BPETokenizer:
    def __init__(self, vocab, merges):
        self.vocab = vocab
        self.merges = merges
        self.pat = re.compile(r" ?[a-zA-Z]+| ?[0-9]+| ?[^\w\s]+|\s+")
        reserved_tokens = ["<pad>", "<unk>", "<s>", "</s>"]
        self.special_to_id = {tok: i for i, tok in enumerate(reserved_tokens)}
        self.id_to_special = {i: tok for i, tok in enumerate(reserved_tokens)}

    def _encode_chunk(self, chunk_str):
        word_ids = [b + 4 for b in chunk_str.encode("utf-8")]
        while len(word_ids) >= 2:
            pairs = [(word_ids[i], word_ids[i + 1]) for i in range(len(word_ids) - 1)]
            
            best_pair = min(
                pairs,
                key=lambda p: self.merges.get(p, float("inf"))
            )
            if best_pair not in self.merges:
                break
            new_id = self.merges[best_pair]
            word_ids = merge_pair_in_word(word_ids, best_pair, new_id)
        return word_ids

    def encode(self, text, add_bos=False, add_eos=False):
        token_ids = []
        for chunk in self.pat.findall(text):
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