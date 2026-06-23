class Vocab:
    def __init__(self, max_vocab=30000):
        self.word2idx = {"<PAD>":0,"<OOV>":1}
        self.idx2word = {0:"<PAD>",1:"<OOV>"}
        self.max_vocab = max_vocab

    def build(self, texts):
        freq = {}
        for text in texts:
            for w in text.lower().split():
                freq[w] = freq.get(w,0)+1

        sorted_words = sorted(freq.items(), key=lambda x:-x[1])
        for i,(w,_) in enumerate(sorted_words[:self.max_vocab-2], start=2):
            self.word2idx[w]=i
            self.idx2word[i]=w

    def encode(self, text):
        return [self.word2idx.get(w,1) for w in text.lower().split()]