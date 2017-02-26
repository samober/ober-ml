import gensim
from gensim.models.word2vec import Vocab

import sys
import argparse
import random

from ober.tokens import TokenDatabase
from ober.documents import DocumentDatabase
        
class Word2Vec:
    def __init__(self, token_database, document_database):
        self.token_database = token_database
        self.document_database = document_database
        
        self.model = gensim.models.Word2Vec(size=token_database.vector_size, window=3, negative=25, sorted_vocab=0)
        
        for token in self.token_database:
            self.model.wv.vocab[token] = Vocab(count=self.token_database.get_freq(token), index=len(self.model.wv.index2word), sample_int=sys.maxint)
            self.model.wv.index2word.append(token)
            
        self.model.finalize_vocab()
        self.model.wv.syn0 = self.token_database.get_vectors()
        
        print self.model.wv.syn0.shape
        print (len(self.token_database), self.token_database.vector_size)
            
    def train(self):
        print "Training ..."
        batch = random.randint(1, self.document_database.num_batches())
        sentences = self.document_database.get_sentences(batch=batch)
        num_sentences = self.document_database.get_batch_stats(batch).total_sentences
        self.model.corpus_count = num_sentences
        self.model.train(sentences, total_examples=num_sentences)
        
        print "Updating vectors ..."
        self.model.init_sims()
        print self.model.wv.syn0.shape
        print (len(self.token_database), self.token_database.vector_size)
        self.token_database.update_vectors(self.model.wv.syn0)
        
        print "Successfully trained %d sentences ..." % num_sentences
        
        print "Saving ..."
        self.token_database.save(vectors_version=self.token_database.vectors_version + 1)
            
    def most_similar(self, word):
        return self.model.most_similar(word)
        
def main():
    parser = argparse.ArgumentParser(description="Runs Word2Vec model on document inventory and outputs trained vectors to token inventory")
    parser.add_argument("--token_db_path", help="path to token inventory", default="data/tokens")
    parser.add_argument("--token_db_version", help="version of token inventory to use", type=int)
    parser.add_argument("--document_db_path", help="path to document inventory", default="data/documents")
    parser.add_argument("--document_db_version", help="version of document inventory to use", type=int)
    parser.add_argument("--document_set", help="document set to use", default="trigrams")
    args = parser.parse_args()
    
    token_db_version = args.token_db_version
    if not token_db_version:
        token_db_version = TokenDatabase.get_latest_version(args.token_db_path)
    document_db_version = args.document_db_version
    if not document_db_version:
        document_db_version = DocumentDatabase.get_latest_version(db_path=args.document_db_path, document_set=args.document_set)
        
    token_database = TokenDatabase.load(db_path=args.token_db_path, version=token_db_version)
    document_database = DocumentDatabase.load(db_path=args.document_db_path, version=document_db_version, document_set=args.document_set)
    
    word2vec = Word2Vec(token_database, document_database)
    word2vec.train()
    
if __name__ == "__main__":
    main()