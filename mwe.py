import pandas as pd
from nltk.tokenize import MWETokenizer


class MWE:
    def __init__(self, vocab_file):
        vocab_df = pd.read_csv(vocab_file)
        vocab_lst = vocab_df["x"].to_list()
        
        self.vocab = vocab_lst
        self.smi2index = dict(zip(vocab_lst, range(len(vocab_lst))))
        self.index2smi = dict(zip(range(len(vocab_lst)), vocab_lst))
        
        self.nltk_vocab = []
        for token in self.vocab:
            self.nltk_vocab.append(tuple([*token]))
        self.nltk_tokenizer = MWETokenizer(self.nltk_vocab, separator='')
    
    def smiles_to_token(self, smiles):
        tokenized_lst = self.nltk_tokenizer.tokenize([*smiles])
        token_ids = [self.smi2index[s] for s in tokenized_lst]
        return token_ids
