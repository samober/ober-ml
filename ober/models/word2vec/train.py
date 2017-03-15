import sys
import argparse
import random

import gensim
from gensim.models.word2vec import Vocab

from ober.tokens import TokenDatabase
from ober.documents import DocumentDatabase
        
class Word2Vec:
    
    """
    
    Trains vectors in a ``TokenDatabase`` using the Word2Vec algorithm and sentences from a ``DocumentDatabase``.
    
    """
    
    def __init__(self, token_database, document_database):
        # set the token and document databases
        self.token_database = token_database
        self.document_database = document_database
        
        # create the gensim model
        self.model = gensim.models.Word2Vec(size=token_database.vector_size, window=3, negative=25, sorted_vocab=0)
        
        # add each token from the token database to the gensim model
        for token in self.token_database:
            self.model.wv.vocab[token] = Vocab(count=self.token_database.get_freq(token), index=len(self.model.wv.index2word), sample_int=sys.maxint)
            self.model.wv.index2word.append(token)
            
        # prepare the model and copy over the existing token vectors
        self.model.finalize_vocab()
        self.model.wv.syn0 = self.token_database.get_vectors()
            
    def train(self):
        
        """
        
        Trains the Word2Vec model on a batch of random sentences from the document inventory.
        
        Updates the vectors in the token inventory once complete and saves them to disk.
        
        """
        
        print "TRAINING ..."
        
        # determine a random batch to train on
        batch = random.randint(1, self.document_database.num_batches())
        
        # get the sentences from that batch
        sentences = self.document_database.get_sentences(batch)
        # get the count of the sentences in the batch from the batch statistics
        num_sentences = self.document_database.get_batch_stats(batch).total_sentences
        
        # update the gensim model values
        self.model.corpus_count = num_sentences
        # train using the sentences
        self.model.train(sentences, total_examples=num_sentences)
        
        print "UPDATING VECTORS ..."
        
        # copy over the trained gensim vectors back to the TokenDatabase
        self.token_database.update_vectors(self.model.wv.syn0)
        
        print "SUCCESSFULLY TRAINED %d SENTENCES ..." % num_sentences
        
        print "SAVING ..."
        
        # save the new vectors to disk and increment the version
        self.token_database.save(new_version=False, new_vectors_version=True)
            
    def most_similar(self, token):
        
        """
        
        Returns the most similar tokens to the token given.
        
        :param token: The target token.
        :type token: str
        :return: A list of the most similar tokens in the database to the target word.
        :rtype: ``List[Tuple[str, float]]``
        
        """
        
        # use the most similar function from TokenDatabase
        return self.model.most_similar(token)
        
def main():
    
    """
    
    This script will run the Word2Vec algorithm and automatically update the vectors in the current TokenDatabase using sentences from the DocumentDatabase.
    
    """
    
    # create an argument parser
    parser = argparse.ArgumentParser(description="Runs Word2Vec model on document inventory and outputs trained vectors to token inventory")
    
    # create arguments
    parser.add_argument("--tokens_path", help="path to token inventory", default="data/tokens")
    parser.add_argument("--tokens_version", help="version of token inventory to use", type=int)
    parser.add_argument("--documents_path", help="path to document inventory", default="data/documents")
    parser.add_argument("--documents_version", help="version of document inventory to use", type=int)
    
    # parse the arguments
    args = parser.parse_args()
        
    # load the token database
    token_database = TokenDatabase.load(db_path=args.tokens_path, version=args.tokens_version)
    # load the document database
    document_database = DocumentDatabase.load(db_path=args.documents_path, version=args.documents_version)
    
    # create the Word2Vec model with token and document databases
    word2vec = Word2Vec(token_database, document_database)
    
    # begin training
    word2vec.train()
    
if __name__ == "__main__":
    main()