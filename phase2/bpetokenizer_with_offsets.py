import re
import pickle

import re

def merge_pair_with_offsets(word, pair, new_id):
    a, b = pair
    new_word = []
    i = 0
    while i < len(word):
        if i < len(word) - 1 and word[i][0] == a and word[i + 1][0] == b:
            new_word.append((new_id, word[i][1] + word[i + 1][1]))
            i += 2
        else:
            new_word.append(word[i])
            i += 1
    return new_word

class BPETokenizerWithOffsets:
    def __init__(self, vocab, merges):
        self.vocab = vocab
        self.merges = merges
        self.exp = re.compile(r" ?[a-zA-Z]+| ?[0-9]+| ?[^\w\s]+|\s+")
        
    def _encode_chunk_with_offsets(self, chunk_str):
        chunk_bytes = chunk_str.encode("utf-8")
        byte_to_char = []
        for char_idx, char in enumerate(chunk_str):
            for _ in range(len(char.encode("utf-8"))):
                byte_to_char.append(char_idx)
                
        word = [(b + 4, [i]) for i, b in enumerate(chunk_bytes)]
        
        while len(word) >= 2:
            pairs = [(word[i][0], word[i + 1][0]) for i in range(len(word) - 1)]
            
            best_pair = min(pairs, key=lambda p: self.merges.get(p, float("inf")))
            if best_pair not in self.merges:
                break
            new_id = self.merges[best_pair]
            word = merge_pair_with_offsets(word, best_pair, new_id)
            
        token_ids = []
        char_offsets = []
        for token_id, byte_indices in word:
            start_char = byte_to_char[byte_indices[0]]
            end_char = byte_to_char[byte_indices[-1]] + 1
            token_ids.append(token_id)
            char_offsets.append((start_char, end_char))
            
        return token_ids, char_offsets

    def encode_with_offsets(self, text):
        token_ids = []
        offsets = []
        for match in self.exp.finditer(text):
            chunk_tokens, chunk_char_offsets = self._encode_chunk_with_offsets(match.group())
            chunk_start = match.start()
            
            for tid, (c_start, c_end) in zip(chunk_tokens, chunk_char_offsets):
                token_ids.append(tid)
                offsets.append((chunk_start + c_start, chunk_start + c_end))
                
        return token_ids, offsets

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
    return BPETokenizerWithOffsets(vocab, merges)

def create_token_labels(offsets, gold_entities):
    labels = ["O"] * len(offsets)
    
    for ent_start, ent_end, ent_label, _ in gold_entities:
        entity_started = False
        for i, (tok_start, tok_end) in enumerate(offsets):
            if tok_end > ent_start and tok_start < ent_end:
                if not entity_started:
                    labels[i] = "B-"+str(ent_label)
                    entity_started = True
                else:
                    labels[i] = "I-"+str(ent_label)
    return labels

def decode_token_labels(offsets, labels, text):
    extracted_entities = []
    current_ent = None

    def save_entity(ent):
        s, e = ent["start"], ent["end"]
        while s < e and text[s].isspace():
            s += 1
        while e > s and text[e - 1].isspace():
            e -= 1
        
        if s < e:
            extracted_entities.append((s, e, ent["type"]))

    for i, (label, (tok_start, tok_end)) in enumerate(zip(labels, offsets)):
        if label.startswith("B-"):
            if current_ent:
                save_entity(current_ent)
            current_ent = {"type": label[2:], "start": tok_start, "end": tok_end}
            
        elif label.startswith("I-"):
            if current_ent and current_ent["type"] == label[2:]:
                current_ent["end"] = tok_end
            else:
                if current_ent:
                    save_entity(current_ent)
                current_ent = {"type": label[2:], "start": tok_start, "end": tok_end}
        else: # "O"
            if current_ent:
                save_entity(current_ent)
                current_ent = None

    if current_ent:
        save_entity(current_ent)

    return extracted_entities