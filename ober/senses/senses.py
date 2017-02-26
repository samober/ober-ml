import numpy as np
import os
from collections import defaultdict
import codecs
	
class SenseDatabase:
	
	UNK_TOKEN = "<UNK>"
	
	def __init__(self, vector_size=300, db_path="data/senses", version=None, vectors_version=None):
		self.vector_size = vector_size
		
		self.sense2index = {}
		self.index2sense = []
		self.token2senses = defaultdict(list)
		
		self.vectors = None
		self.norm_vectors = None
		
		self.add_sense(SenseDatabase.UNK_TOKEN)
		self.unk_token = SenseDatabase.UNK_TOKEN
		self.unk_value = self.sense2index[self.unk_token]
		
		self.db_path = db_path
		self.version = version
		self.vectors_version = vectors_version
		
	def add_sense(self, sense):
		"""
		
		Add a sense to the sense inventory.
		
		Senses are in the form ``<token>#<sense_id>``
		
		Example:
			
		.. code-block:: python
		
			# token = 'flying'
			# sense_id = 2
			sense_database.add_sense("flying#2")
		
		:param sense: The sense to add.
		:type sense: str
		
		"""
		if not sense in self.sense2index:
			self.sense2index[sense] = len(self.index2sense)
			self.index2sense.append(sense)
			self.token2senses[sense.rpartition("#")[0]].append(sense)
			
	def add_senses(self, senses):
		for sense in senses:
			self.add_sense(sense)
			
	def encode_sense(self, sense):
		if sense not in self.sense2index:
			return self.unk_value
		return self.sense2index[sense]

	def encode(self, senses):
		return [ self.encode_sense(sense) for sense in senses ]
		
	def decode_sense(self, index):
		return self.index2sense[index]

	def decode(self, indices):
		return [ self.decode_sense(index) for index in indices ]

	def senses_for_token(self, token):
		return self.token2senses[token]
		
	def generate_random_vectors(self):
		self.vectors = np.float32(np.random.uniform(-0.5/self.vector_size, 0.5/self.vector_size, size=(len(self), self.vector_size)))
		
	def generate_zero_vectors(self):
		self.vectors = np.zeros((len(self), self.vector_size), dtype=np.float32)
		
	def load_vectors(self, vectors_path):
		self.vectors = np.load(vectors_path)
		
	def get_vectors(self):
		return self.vectors

	def update_vectors(self, vectors):
		assert vectors.shape == (len(self), self.vector_size)
		self.vectors = vectors
			
	def save_vectors(self, db_path="data/senses", version=1, vectors_version=1):
		np.save(SenseDatabase.get_vectors_path(db_path, version, vectors_version), self.get_vectors())
		
	def get_norm_vectors(self, calculate=False):
		if calculate or self.norm_vectors is None:
			self.norm_vectors = self.get_vectors() / np.sqrt(np.sum(np.square(self.get_vectors()), axis=1, keepdims=True))
		return self.norm_vectors
		
	def most_similar(self, sense, num_similar=12):
		if sense not in self.sense2index:
			return []
		norm_vectors = self.get_norm_vectors()
		similar = np.dot(norm_vectors, norm_vectors[self.sense2index[sense]])
		return [ (self.decode_sense(index), similar[index]) for index in np.argsort(-similar)[1:num_similar+1] ]
			
	def __len__(self):
		return len(self.index2sense)
		
	def __contains__(self, sense):
		return sense in self.sense2index
		
	def __iter__(self):
		for sense in self.index2sense:
			yield sense
		
	@staticmethod
	def load(db_path="data/senses", version=None, vectors_version=None):
		if not version:
			version = SenseDatabase.get_latest_version(db_path)
		if not vectors_version:
			vectors_version = SenseDatabase.get_latest_vectors_version(db_path, version)
		db = SenseDatabase()
		inventory_path = SenseDatabase.get_inventory_path(db_path, version)
		with codecs.open(inventory_path, "rb", "utf-8") as f:
			db.vector_size = int(f.readline())
			for line in f:
				db.add_sense(line.strip())
		db.load_vectors(SenseDatabase.get_vectors_path(db_path, version, vectors_version))
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
		SenseDatabase.ensure_senses_dir(db_path, version)
		inventory_path = SenseDatabase.get_inventory_path(db_path=db_path, version=version)
		with codecs.open(inventory_path, "wb", "utf-8") as f:
			f.write("%d\n" % self.vector_size)
			for sense in self.index2sense:
				f.write("%s\n" % sense)
		self.save_vectors(db_path, version, vectors_version)
		self.db_path = db_path
		self.version = version
		self.vectors_version = vectors_version
			
	@staticmethod
	def get_latest_version(db_path="data/senses"):
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
	def get_latest_vectors_version(db_path="data/senses", version=None):
		if not version:
			version = SenseDatabase.get_latest_version(db_path)
		max_version = 0
		vectors_dir = os.path.join(db_path, "%05d" % version, "vectors")
		if os.path.exists(vectors_dir):
			for f in os.listdir(vectors_dir):
				if f.endswith(".npy"):
					v = 0
					try:
						v = int(f.split(".")[0])
					except:
						v = 0
					if v > max_version:
						max_version = v
			return max_version
		return 0
		
	@staticmethod
	def get_inventory_path(db_path="data/senses", version=1):
		return os.path.join(db_path, "%05d" % version, "inventory.vocab")

	@staticmethod
	def get_vectors_path(db_path="data/senses", version=1, vectors_version=1):
		return os.path.join(db_path, "%05d" % version, "vectors", "%04d.npy" % vectors_version)
		
	@staticmethod
	def ensure_senses_dir(db_path="data/senses", version=1):
		version_dir = os.path.join(db_path, "%05d" % version)
		if not os.path.exists(version_dir):
			os.makedirs(version_dir)
		vectors_dir = os.path.join(version_dir, "vectors")
		if not os.path.exists(vectors_dir):
			os.makedirs(vectors_dir)