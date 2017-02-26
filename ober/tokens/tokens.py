import numpy as np
import os
from collections import defaultdict
import codecs
from tokens_inner import export_distributed_graph
    
class TokenDatabase:
    
    PAD_TOKEN = "<PAD>"
    UNK_TOKEN = "<UNK>"
    START_VOCAB = [ PAD_TOKEN, UNK_TOKEN ]
    
    def __init__(self, vector_size=300, db_path="data/tokens", version=None, vectors_version=None):
        self.vector_size = vector_size
        
        self.token2index = {}
        self.index2token = []
        self.token_freq = defaultdict(int)
        
        self.vectors = None
        self.norm_vectors = None
        
        self.add_tokens(TokenDatabase.START_VOCAB, counts=[1000, 1000])
            
        self.pad_token = TokenDatabase.PAD_TOKEN
        self.unk_token = TokenDatabase.UNK_TOKEN
        self.pad_value = self.token2index[TokenDatabase.PAD_TOKEN]
        self.unk_value = self.token2index[TokenDatabase.UNK_TOKEN]
        
        self.db_path = None
        self.version = version
        self.vectors_version = vectors_version
        
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
        if not counts:
            counts = [1] * len(tokens)
        for index, token in enumerate(tokens):
            self.add_token(token, count=counts[index])
            
    def encode_token(self, token):
        if token not in self.token2index:
            return self.unk_value
        return self.token2index[token]

    def encode(self, tokens):
        return [ self.encode_token(token) for token in tokens ]
        
    def decode_token(self, index):
        return self.index2token[index]

    def decode(self, indices):
        return [ self.decode_token(index) for index in indices ]
        
    def get_freq(self, token):
        return self.token_freq[token]
        
    def generate_random_vectors(self):
        self.vectors = np.float32(np.random.uniform(-0.5/self.vector_size, 0.5/self.vector_size, size=(len(self), self.vector_size)))
        
    def load_vectors(self, vectors_path):
        self.vectors = np.load(vectors_path)
        
    def get_vectors(self):
        return self.vectors

    def update_vectors(self, vectors):
        assert vectors.shape == (len(self), self.vector_size)
        self.vectors = vectors
            
    def save_vectors(self, db_path="data/tokens", version=1, vectors_version=1):
        np.save(TokenDatabase.get_vectors_path(db_path, version, vectors_version), self.get_vectors())
        
    def get_norm_vectors(self, calculate=False):
        if calculate or self.norm_vectors is None:
            self.norm_vectors = self.get_vectors() / np.sqrt(np.sum(np.square(self.get_vectors()), axis=1, keepdims=True))
        return self.norm_vectors
        
    def most_similar(self, token, num_similar=12):
        if token not in self.token2index:
            return []
        norm_vectors = self.get_norm_vectors()
        similar = np.dot(norm_vectors, norm_vectors[self.token2index[token]])
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
        if not version:
            version = TokenDatabase.get_latest_version(db_path)
        if not vectors_version:
            vectors_version = TokenDatabase.get_latest_vectors_version(db_path, version)
        db = TokenDatabase()
        counts_path = TokenDatabase.get_counts_path(db_path=db_path, version=version)
        with codecs.open(counts_path, "rb", "utf-8") as f:
            db.vector_size = int(f.readline())
            for line in f:
                try:
                    token = line.split("\t")[0]
                    count = int(line.split("\t")[1])
                    db.add_token(token, count=count)
                except:
                    continue
        db.load_vectors(TokenDatabase.get_vectors_path(db_path=db_path, version=version, vectors_version=vectors_version))
        db.db_path = db_path
        db.version = version
        db.vectors_version = vectors_version
        return db
                
    def save(self, db_path=None, version=None, vectors_version=None):
        if not db_path:
            db_path = self.db_path
        if not version:
            version = self.version
        if not vectors_version:
            vectors_version = self.vectors_version
        TokenDatabase.ensure_tokens_dir(db_path, version)
        counts_path = TokenDatabase.get_counts_path(db_path=db_path, version=version)
        with codecs.open(counts_path, "wb", "utf-8") as f:
            f.write("%d\n" % self.vector_size)
            for index, token in enumerate(self.index2token):
                f.write("%s\t%d\n" % (token, self.token_freq[token]))
        self.save_vectors(db_path, version, vectors_version)
        self.db_path = db_path
        self.version = version
        self.vectors_version = vectors_version
            
    def export_distributed_graph(self, db_path=None, version=None, graph_version=1, n=200, batch_size=250):
        if not db_path:
            db_path = self.db_path
        if not version:
            version = self.version
        graph_path = TokenDatabase.get_graph_path(db_path, version, graph_version)
        export_distributed_graph(self.get_norm_vectors(), graph_path, n=n, batch_size=batch_size)
                    
    @staticmethod
    def get_latest_version(db_path="data/tokens"):
        max_version = 0
        for directory in os.listdir(db_path):
            version = 0
            try:
                version = int(directory)
            except:
                version = 0
            if version > max_version:
                max_version = version
        return max_version
        
    @staticmethod
    def get_latest_vectors_version(db_path="data/tokens", version=None):
        if not version:
            version = TokenDatabase.get_latest_version(db_path)
        max_version = 0
        for f in os.listdir(os.path.join(db_path, "%05d" % version, "vectors")):
            if f.endswith(".npy"):
                v = 0
                try:
                    v = int(f.split(".")[0])
                except:
                    v = 0
                if v > max_version:
                    max_version = v
        return max_version
        
    @staticmethod
    def get_latest_graph_version(db_path="data/tokens", version=None):
        if not version:
            version = TokenDatabase.get_latest_version(db_path)
        max_version = 0
        for f in os.listdir(os.path.join(db_path, "%05d" % version, "graphs")):
            if f.endswith(".dt"):
                v = 0
                try:
                    v = int(f.split(".")[0])
                except:
                    v = 0
                if v > max_version:
                    max_version = v
        return max_version
        
    @staticmethod
    def get_counts_path(db_path="data/tokens", version=1):
        return os.path.join(db_path, "%05d" % version, "counts.vocab")

    @staticmethod
    def get_vectors_path(db_path="data/tokens", version=1, vectors_version=1):
        return os.path.join(db_path, "%05d" % version, "vectors", "%04d.npy" % vectors_version)

    @staticmethod
    def get_graph_path(db_path="data/tokens", version=1, graph_version=1):
        return os.path.join(db_path, "%05d" % version, "graphs", "%04d.dt" % graph_version)
        
    @staticmethod
    def get_latest_graph_path(db_path):
        tokens_version = TokenDatabase.get_latest_version(db_path=db_path)
        graph_version = TokenDatabase.get_latest_graph_version(db_path=db_path)
        if graph_version != 0:
            return TokenDatabase.get_graph_path(db_path=db_path, version=tokens_version, graph_version=graph_version)
        return None
        
    @staticmethod
    def ensure_tokens_dir(db_path="data/tokens", version=1):
        version_dir = os.path.join(db_path, "%05d" % version)
        if not os.path.exists(version_dir):
            os.makedirs(version_dir)
        vectors_dir = os.path.join(version_dir, "vectors")
        if not os.path.exists(vectors_dir):
            os.makedirs(vectors_dir)
        graphs_dir = os.path.join(version_dir, "graphs")
        if not os.path.exists(graphs_dir):
            os.makedirs(graphs_dir)
        