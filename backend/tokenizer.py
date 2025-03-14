from collections import Counter

class Tokenizer:
    def __init__(self, texts, max_vocab_size=None):
        self.word_index = {"<pad>": 0, "<sos>": 1, "<eos>": 2, "<unk>": 3}
        self.index_word = {0: "<pad>", 1: "<sos>", 2: "<eos>", 3: "<unk>"}
        self.build_vocab(texts, max_vocab_size)
    def build_vocab(self, texts, max_vocab_size):
        word_counts = Counter()
        for text in texts:
            for word in text.split():
                word_counts[word] += 1
        vocab_size = len(self.word_index)
        for word, _ in word_counts.most_common(max_vocab_size):
            if word not in self.word_index:
                self.word_index[word] = vocab_size
                self.index_word[vocab_size] = word
                vocab_size += 1

    def texts_to_sequences(self, texts):
        sequences = []
        for text in texts:
            sequence = []
            sequence.append(self.word_index["<sos>"])
            for word in text.split():
                if word in self.word_index:
                    sequence.append(self.word_index[word])
                else:
                    sequence.append(self.word_index["<unk>"])
            sequence.append(self.word_index["<eos>"])
            sequences.append(sequence)
        return sequences

    def sequences_to_texts(self, sequences):
        texts = []
        for sequence in sequences:
            text = []
            for idx in sequence:
                if idx in self.index_word and idx not in [0, 1, 2]:
                    text.append(self.index_word[idx])
            texts.append(" ".join(text))
        return texts
    def size(self):
        return len(self.word_index)