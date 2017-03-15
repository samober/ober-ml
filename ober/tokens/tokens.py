import numpy as np
import os
from collections import defaultdict
import codecs
from tokens_inner import export_distributed_graph
from ober.data import VersionedFile
    
class TokenDatabase:
    """
    
    A ``TokenDatabase`` is an interface to the token inventory. It stores all tokens, token frequencies, and token vectors.
    
    It allows for fast and simple encoding and decoding of token sequences to and from integer ids.
    
    Encoding:
        
    .. code-block:: python
    
        sentence = [ "the", "man", "sat", "on", "the", "bench" ]
        token_database.encode(sentence)
        # [ 10, 356, 5832, 9432, 10, 28473 ]
        
    Decoding:
        
    .. code-block:: python
    
        sentence = [ 10, 356, 5832, 9432, 10, 28473 ]
        token_database.decode(sentence)
        # [ "the", "man", "sat", "on", "the", "bench" ]
    
    """
    
    PAD_TOKEN = "<PAD>"
    """ Special token ``<PAD>`` for padding token sequences. """
    UNK_TOKEN = "<UNK>"
    """ Special token ``<UNK>`` for unknown values during encoding/decoding. """
    
    def __init__(self, vector_size=300, db_path="data/tokens", version=None, vectors_version=None):
        self.vector_size = vector_size
        
        self.token2index = {}
        self.index2token = []
        self.token_freq = defaultdict(int)
        
        self.vectors = None
        self.norm_vectors = None
        
        self.add_tokens([ TokenDatabase.PAD_TOKEN, TokenDatabase.UNK_TOKEN ], counts=[1000, 1000])
            
        self.pad_token = TokenDatabase.PAD_TOKEN
        self.unk_token = TokenDatabase.UNK_TOKEN
        self.pad_value = self.token2index[TokenDatabase.PAD_TOKEN]
        self.unk_value = self.token2index[TokenDatabase.UNK_TOKEN]
        
        self.db_path = db_path
        self.version = version
        self.vectors_version = vectors_version
        
        self.file_base = None
        self.version_vectors = None
        
        # load file systems
        self._load_file_system(version=self.version)
        self._load_vectors_file_system(vectors_version=self.vectors_version)
        
    def _get_counts_path(self):
        return self.file_base.get_file_path(self.version, "counts.vocab")
        
    def _get_vectors_file(self, vectors_version):
        if not vectors_version:
            return self.version_vectors.get_latest_file_path("vectors.npy")
        return self.version_vectors.get_file_path(vectors_version, "vectors.npy")
            
    def _load_file_system(self, version=None, new_version=False):
        # create the file system
        self.file_base = VersionedFile(self.db_path)
        self.version = version
        # if no version specified use the latest available version
        if not self.version:
            self.version = self.file_base.get_latest_version()
        # update version
        if new_version or self.version == 0:
            self.version = self.file_base.create_latest_version()
            
    def _load_vectors_file_system(self, vectors_version=None, new_version=False):
        # create the file system
        self.version_vectors = VersionedFile(self.file_base.get_file_path(self.version, "vectors"), version_num_length=4)
        self.vectors_version = vectors_version
        # if no vectors version is specified use the latest available version
        if not self.vectors_version:
            self.vectors_version = self.version_vectors.get_latest_version()
        # update vectors version
        if new_version or self.vectors_version == 0:
            self.vectors_version = self.version_vectors.create_latest_version()
        
    def add_token(self, token, count=1):
        """
        
        Add a token to this token database and increment it's count.
        
        :param token: The token to add.
        :type token: str
        :param count: The amount to increment the token's frequency by.
        :type count: int
   
        """
        if token.strip() == "":
            return
        if not token in self.token2index:
            self.token2index[token] = len(self.index2token)
            self.index2token.append(token)
        self.token_freq[token] += count
        
    def add_tokens(self, tokens, counts=None):
        """
        
        Adds all tokens from the list ``tokens`` to the token inventory.
        
        :param tokens: List of tokens to add.
        :type tokens: ``list[str]``
        :param counts: *(optional)* List of frequency counts for each token.
        :type counts: ``list[int]``
        
        """
        if not counts:
            counts = [1] * len(tokens)
        for index, token in enumerate(tokens):
            self.add_token(token, count=counts[index])
            
    def encode_token(self, token):
        """
        
        Returns the token id for the given token, otherwise returns the id for the special <UNK> token.
        
        :param token: The token to encode.
        :type token: str
        :return: The id of the token.
        :rtype: int
        
        """
        if token not in self.token2index:
            return self.unk_value
        return self.token2index[token]

    def encode(self, tokens):
        """
        
        Takes an array of plain text tokens and returns an array of their corresponding token ids.
        
        :param tokens: An array of plain text tokens.
        :type tokens: ``List[str]``
        :return: An array of token indices.
        :rtype: ``List[int]``
        
        """
        return [ self.encode_token(token) for token in tokens ]
        
    def decode_token(self, index):
        """
        
        Returns the plain text of the token with index ``index``.
        
        :param index: The index of the token.
        :type index: int
        :return: The token text of the token with index ``index``.
        :rtype: str
        
        """
        return self.index2token[index]

    def decode(self, indices):
        """
        
        Takes an array of indices and returns an array of their corresponding plain text tokens.
        
        :param indices: An array of indices representing token ids.
        :type indices: ``List[int]``
        :return: An array of token texts.
        :rtype: ``List[str]``
        
        """
        return [ self.decode_token(index) for index in indices ]
        
    def get_freq(self, token):
        """
        
        Returns the frequency the specified token has occured in the training corpus.
        
        :param token: The target token.
        :type token: str
        :return: The token's frequency.
        :rtype: int
        
        """
        return self.token_freq[token]
        
    def generate_random_vectors(self):
        """
        
        Generates random token vectors for all tokens using a uniform distribution.
        
        Use this for initializing vectors for new databases or before transfering old vectors to a new database.
        
        """
        self.vectors = np.float32(np.random.uniform(-0.5/self.vector_size, 0.5/self.vector_size, size=(len(self), self.vector_size)))
        
    def load_vectors(self, vectors_version=None):
        """
        
        Loads the vectors from disk.
        
        If vectors_version is specified this will load that version of the vectors, otherwise it will load the latest.
        
        :param vectors_version: The version of the vectors to load. (default is latest)
        :type vectors_version: bool
        
        """
        self.vectors = np.load(self._get_vectors_file(vectors_version))
        
    def get_vectors(self):
        """
        
        Returns the token vectors for this token database.
        
        :return: A Numpy array of the token vectors for this database.
        :rtype: ``np.ndarray``
        
        """
        return self.vectors

    def update_vectors(self, vectors):
        """
        
        Updates the current token vectors with new ones.
        
        The new vectors array must be of the shape (vocab length, vector size)
        
        :param vectors: The Numpy array of new token vectors.
        :type vectors: ``np.ndarray``
        
        """
        assert vectors.shape == (len(self), self.vector_size)
        self.vectors = vectors
            
    def save_vectors(self, new_version=True):
        """
        
        Saves the vectors to disk.
        
        If new_version is True, the version number will automatically be incremented when saving, otherwise the current database vectors version will be used.
        
        :param new_version: True if version should be incremented, False to use the current vector version.
        :type new_version: bool
        
        """
        # create a new version if flagged
        if new_version:
            self.vectors_version = self.version_vectors.create_latest_version()
        # save the numpy array to disc
        np.save(self._get_vectors_file(self.vectors_version), self.get_vectors())
        
    def get_norm_vectors(self, calculate=False):
        """
        
        Returns the normalized token vectors for this database.
        
        :param calculate: If True, the normalized vectors will be recalculated before being returned. If False, a cached value may be returned.
        :type calculate: bool
        :return: A Numpy array of all the normalized token vectors in this database.
        :rtype: ``np.ndarray``
        
        """
        if calculate or self.norm_vectors is None:
            self.norm_vectors = self.get_vectors() / np.sqrt(np.sum(np.square(self.get_vectors()), axis=1, keepdims=True))
        return self.norm_vectors
        
    def most_similar(self, token, num_similar=12):
        """
        
        Returns the most similar tokens based on the cosine similarity between token vectors.
        
        :param token: The target token.
        :type token: str
        :param num_similar: The number of results to return.
        :type num_similar: int
        :return: A list of the top ``num_similar`` tokens to the target token.
        :rtype: ``List[Tuple[str, float]]``
        
        """
        # if not a known token, return an empty list
        if token not in self.token2index:
            return []
        # get the normalized vectors
        norm_vectors = self.get_norm_vectors()
        # take the cosine similarity of the target token's vector to all the other tokens' vectors
        similar = np.dot(norm_vectors, norm_vectors[self.token2index[token]])
        # sort and decode the indices of the top similar
        return [ (self.decode_token(index), similar[index]) for index in np.argsort(-similar)[1:num_similar+1] ]
            
    def __len__(self):
        return len(self.index2token)
        
    def __contains__(self, token):
        return token in self.token2index
        
    def __iter__(self):
        for token in self.index2token:
            yield token
            
    @staticmethod
    def load(db_path="data/tokens", version=None, vectors_version=None):
        """
        
        Loads a specific version of the token database from disk.
        
        :param db_path: The path to the directory of the token inventory.
        :type db_path: str
        :param version: The version number of the database to load. (Defaults to latest)
        :type version: int
        :param vectors_version: The version of the token vectors to load. (Defaults to latest)
        :type vectors_version: int
        :return: A ``TokenDatabase`` object representing the token database of the specified version.
        :rtype: ``TokenDatabase``
        
        """
        # create the database
        db = TokenDatabase(db_path=db_path, version=version, vectors_version=vectors_version)
        # get the path to the counts file
        counts_path = db._get_counts_path()
        with codecs.open(counts_path, "rb", "utf-8") as f:
            # vector size is first line
            db.vector_size = int(f.readline())
            # read in tokens/counts line by line (file format: {word}<TAB>{count}\n)
            for line in f:
                try:
                    token = line.split("\t")[0]
                    count = int(line.split("\t")[1])
                    db.add_token(token, count=count)
                except:
                    # something went wrong when parsing, just skip
                    continue
        # load the vectors from file
        db.load_vectors()
        return db
                
    def save(self, db_path=None, new_version=True, new_vectors_version=True):
        """
        
        Saves this token database's vocabulary counts and word vectors to disk.
        
        :param db_path: The path for the token inventory (defaults to the one this token database was initialized with.
        :type db_path: str
        :param new_version: If True, increment the version number and save as the latest version.
        :type new_version: bool
        :param new_vectors_version: If True, increment the vectors version and save the vectors as the latest version.
        :type new_vectors_version: bool
        
        """
        # if db_path is new, completely re-initialize everything
        # if there is a new version, reload the file system and re-initialize the entire vectors file system
        # if there is only a new vectors version, just reload the vectors file system
        if db_path is not None and self.db_path != db_path:
            # update db path
            self.db_path = db_path
            # completely reload file system
            self._load_file_system(new_version=new_version)
            # complete reload vectors file system
            self._load_vectors_file_system(new_version=new_vectors_version)
        elif new_version:
            # reload the file system
            self._load_file_system(version=self.version, new_version=True)
            # completely reload vectors file system
            self._load_vectors_file_system(new_version=new_vectors_version)
        elif new_vectors_version:
            # reload the vectors file system
            self._load_vectors_file_system(vectors_version=self.vectors_version, new_version=True)
        
        # get the path to the counts file and write counts
        counts_path = self._get_counts_path()
        with codecs.open(counts_path, "wb", "utf-8") as f:
            f.write("%d\n" % self.vector_size)
            for index, token in enumerate(self.index2token):
                f.write("%s\t%d\n" % (token, self.token_freq[token]))
        # save vectors (new version is taken care of above so no need here)
        self.save_vectors(new_version=False)
            
    def export_distributed_graph(self, n=200, batch_size=250):
        """
        
        Exports a file with the data representing an undirected similarity graph of the entire vocabulary.
        
        Each token has up to 200 of it's closest neighbors as edges, with the edge weights equal to the cosine similarity between the two token's normalized vectors.
        
        :param n: The maximum number of edges each token may have.
        :type n: int
        :param batch_size: The size of the processing batches for the worker threads.
        :type batch_size: int
        
        """
        # load up the graph file system
        graph_files = VersionedFile(self.file_base.get_file_path(self.version, "graphs"), version_num_length=4)
        # create a new version
        graph_files.create_latest_version()
        # get graph data file path
        graph_path = graph_files.get_latest_file_path("graph.dt")
        export_distributed_graph(self.get_norm_vectors(), graph_path, n=n, batch_size=batch_size)
        