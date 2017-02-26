import os
import bz2
import json
import codecs
import random

class DocumentDatabase:
	"""
	
	A ``DocumentDatabase`` manages a set of documents in the document inventory with a specified version and subset.
	
	For example, you could have a ``DocumentDatabase`` of all documents in a document set called "bigrams", which store documents after being run through a bigram phraser. 
	
	You could also have multiple versions of your inventory within this set. For example, if you add a sizeable number of new documents and update your phraser, you could then rephrase all your documents into version 2 of the bigrams document set.
	
	This is illustrated below:
		
	.. code-block:: python
	
		from ober.documents import DocumentDatabase
		
		# load a document database located at 'data/documents'
		# from version 1 of the document set "bigrams"
		document_database = DocumentDatabase.load(db_path="data/documents", version=1, document_set="bigrams")
		
		### DO SOME WORK
		
		# save the new documents to version 2 of the document set "bigrams"
		document_database.save("data/documents", version=2, document_set="bigrams")
		
	Documents are stored as JSON lines files by batches:
		
	.. code-block:: python
		
		<inventory_directory>
			<document_set>
				<version_id>
					<batch_id>.jl.bz2
					<batch_id>.stats.json
					[...]
				[...]
			<document_set>
				<version_id>
					<batch_id>.jl.bz2
					<batch_id>.stats.json
					[...]
				[...]
			[...]
			
	Each document has the format:
		
	.. code-block:: python
	
		{
			title: <document_title>,
			paragraphs: [
				{
					sentences: [
						{
							tokens: [ "the", "sentence", "tokens", [...] ]
						},
						[...]
					]
				},
				[...]
			]
		}
	
	"""
	
	def __init__(self, db_path="data/documents", version=None, document_set="trigrams"):
		self.db_path = db_path
		self.version = version
		self.document_set = document_set
		self.batch_stats = {}
		
	def _get_batches(self):
		return DocumentDatabase.get_batches(self.db_path, self.version, self.document_set)
		
	def _get_batch_files(self):
		return DocumentDatabase.get_batch_files(self.db_path, self.version, self.document_set)
		
	def _get_batch_file(self, batch=1):
		return DocumentDatabase.get_batch_file(self.db_path, self.version, self.document_set, batch)
		
	def _get_batch_stat_file(self, batch=1):
		return DocumentDatabase.get_batch_stat_file(self.db_path, self.version, self.document_set, batch)
			
	def get_documents(self, batch=None):
		"""
		
		Returns an iterator of documents being read from disk.
		
		:param batch:	* None - iterates over all batches in this inventory
						* batch_index - iterates over the batch at the specified index
						* 'random' - picks a batch at random and iterates through it
		:type batch: [None|int|'random']
		:return: An iterator of JSON documents.
		:rtype: ``Iterator[dict]``
		
		"""
		files = None
		if not batch:
			files = self._get_batch_files()
		elif batch == "random":
			batches = self._get_batches()
			files = [ self._get_batch_file(batch=random.randint(1, len(batches))) ]
		else:
			files = [ self._get_batch_file(batch=batch) ]
		for f in files:
			with bz2.BZ2File(f, "r") as infile:
				for line in infile:
					yield json.loads(line)
					
	def get_paragraphs(self, batch=None):
		"""
		
		Returns an interator of paragraphs being read from disk.
		
		:param batch:	* None - iterates over all batches in this inventory
						* batch_index - iterates over the batch at the specified index
						* 'random' - picks a batch at random and iterates through it
		:type batch: [None|int|'random']
		:return: An iterator of JSON paragraphs.
		:rtype: ``Iterator[dict]``
		
		"""
		for document in self.get_documents(batch):
			for paragraph in document["paragraphs"]:
				yield paragraph["sentences"]
		
	def get_sentences(self, batch=None):
		"""
		
		Returns an interator of sentences being read from disk.
		
		:param batch:	* None - iterates over all batches in this inventory
						* batch_index - iterates over the batch at the specified index
						* 'random' - picks a batch at random and iterates through it
		:type batch: [None|int|'random']
		:return: An iterator of tokenized sentences.
		:rtype: ``Iterator[list[str]]``
		
		"""
		for paragraph in self.get_paragraphs(batch):
			for sentence in paragraph:
				yield sentence["tokens"]
				
	def get_batch_stats(self, batch):
		"""
		
		Returns the ``BatchStats`` for a specific batch.
		
		:param batch: Batch index.
		:type batch: int
		:return: ``BatchStats`` for the batch.
		:rtype: ``BatchStats``
		
		"""
		return self.batch_stats[batch]
		
	def num_batches(self):
		"""
		
		Returns the number of batches in the current document inventory.
		
		:return: Total number of batches.
		:rtype: int
		
		"""
		return len(self.batch_stats)
		
	def get_total_sentences(self):
		"""
		
		Convenience method that sums up all the sentences across all batches.
		
		:return: Total number of sentences in the document inventory.
		:rtype: int
		
		"""
		total_sentences = 0
		for batch in self.batch_stats:
			total_sentences += self.batch_stats[batch].total_sentences
		return total_sentences
		
	def add_documents(self, documents):
		"""
		
		Adds :class:`documents` to the document inventory, writing to disk in batches of 500,000.
		
		:param documents: An iterator of JSON documents.
		:type documents: ``Iterator[dict]``
		
		"""
		DocumentDatabase.ensure_documents_dir(self.db_path, self.version, self.document_set)
		more_documents = True
		while more_documents:
			batch = len(self.batch_stats) + 1
			sentences_count = 0
			batch_file = self._get_batch_file(batch) + ".temp"
			try:
				document = documents.next()
				with bz2.BZ2File(batch_file, "w") as outfile:
					for i in range(500000):
						for paragraph in document["paragraphs"]:
							sentences_count += len(paragraph["sentences"])
						outfile.write("%s\n" % json.dumps(document))
						if i != 499999:
							document = documents.next()
			except StopIteration:
				more_documents = False
			if sentences_count > 0:
				batch_stat_file = self._get_batch_stat_file(batch)
				self.batch_stats[batch] = BatchStats(sentences_count)
				with codecs.open(batch_stat_file, "wb", "utf-8") as outfile:
					outfile.write(json.dumps(self.batch_stats[batch].to_json()))
				os.rename(batch_file, batch_file[:-5]) # take away .temp so it become valid batch
	
	@staticmethod
	def load(db_path="data/documents", version=None, document_set="trigrams"):
		"""
		
		Loads a document inventory from ``docuemnt_set`` at ``db_path`` with version ``version``.
		
		:param db_path: The path of the document inventory.
		:type db_path: str
		:param version: The version of the document inventory.
		:type version: int
		:param document_set: The document set to load from.
		:type document_set: str
		:return: A ``DocumentDatabase`` object representing the document inventory.
		:rtype: ``DocumentDatabase``
		
		"""
		if not version:
			version = DocumentDatabase.get_latest_version(db_path, document_set)
		db = DocumentDatabase(db_path, version, document_set)
		for batch in db._get_batches():
			stats_file = db._get_batch_stat_file(batch)
			stats_json = json.loads(codecs.open(stats_file, "rb", "utf-8").read())
			db.batch_stats[batch] = BatchStats(stats_json["total_sentences"])
		return db
	
	@staticmethod
	def get_latest_version(db_path="data/documents", document_set="trigrams"):
		"""
		
		Returns the highest version of the document inventory at ``db_path`` from set ``document_set``.
		
		:param db_path: The path of the document inventory.
		:type db_path: str
		:param document_set: The document set to look at.
		:type document_set: str
		:return: The latest version number from the document inventory.
		:rtype: int
		
		"""
		max_version = 0
		for directory in os.listdir(os.path.join(db_path, document_set)):
			version = 0
			try:
				version = int(directory)
			except:
				version = 0
			if version > max_version:
				max_version = version
		return max_version
		
	@staticmethod
	def get_documents_directory(db_path, version, document_set):
		"""
		
		Returns the specific directory for the document inventory at ``db_path``.
		
		:param db_path: The path to the document inventory.
		:type db_path: str
		:param version: The version of the document inventory.
		:type version: int
		:param document_set: The target document set.
		:type document_set: str
		:return: Relative path to document inventory directory.
		:rtype: str
		
		"""
		return os.path.join(db_path, document_set, "%05d" % version)
		
	@staticmethod
	def get_batches(db_path, version, document_set):
		"""
		
		Returns the indices of the batches of a specific document inventory.
		
		:param db_path: The path to the document inventory.
		:type db_path: str
		:param version: The version of the document inventory.
		:type version: int
		:param document_set: The target document set.
		:type document_set: str
		:return: A list of batch indices for the document inventory.
		:rtype: ``list[int]``
		
		"""
		documents_directory = DocumentDatabase.get_documents_directory(db_path, version, document_set)
		return [ int(f.split(".")[0]) for f in os.listdir(documents_directory) if f.endswith(".jl.bz2") ]
		
	@staticmethod
	def get_batch_file(db_path, version, document_set, batch):
		"""
		
		Returns the path to a batch file in a document inventory.
		
		:param db_path: The path to the document inventory.
		:type db_path: str
		:param version: The version of the document inventory.
		:type version: int
		:param document_set: The target document set.
		:type document_set: str
		:param batch: The batch index of the document inventory.
		:type batch: int
		:return: Relative path to batch file for the document inventory.
		:rtype: str
		
		"""
		return os.path.join(db_path, document_set, "%05d" % version, "%04d.jl.bz2" % batch)
		
	@staticmethod
	def get_batch_stat_file(db_path, version, document_set, batch):
		"""
		
		Returns the path to a stat file for a batch in a document inventory.
		
		:param db_path: The path to the document inventory.
		:type db_path: str
		:param version: The version of the document inventory.
		:type version: int
		:param document_set: The target document set.
		:type document_set: str
		:param batch: The batch index of the document inventory.
		:type batch: int
		:return: Relative path to the stat file for the batch in a document inventory.
		:rtype: str
		
		"""
		return os.path.join(db_path, document_set, "%05d" % version, "%04d.stats.json" % batch)
		
	@staticmethod
	def get_batch_files(db_path, version, document_set):
		"""
		
		Returns a list of all batch file paths for a specific document inventory.
		
		:param db_path: The path to the document inventory.
		:type db_path: str
		:param version: The version of the document inventory.
		:type version: int
		:param document_set: The target document set.
		:type document_set: str
		:return: A list of batch file paths for the document inventory.
		:rtype: ``list[str]``
		
		"""
		return [ DocumentDatabase.get_batch_file(db_path, version, document_set, batch) for batch in DocumentDatabase.get_batches(db_path, version, document_set) ]
		
	@staticmethod
	def ensure_documents_dir(db_path, version, document_set):
		"""
		
		Makes sure the proper directories are in place in order to save the document inventory.
		
		:param db_path: The path to the document inventory.
		:type db_path: str
		:param version: The version of the document inventory.
		:type version: int
		:param document_set: The target document set.
		
		"""
		documents_directory = DocumentDatabase.get_documents_directory(db_path, version, document_set)
		if not os.path.exists(documents_directory):
			os.makedirs(documents_directory)
		
class BatchStats:
	"""
	
	Stores information about a specific batch in a document inventory.
	
	total_sentences *(int)*
		Total number of sentences in the batch.
	
	"""
	
	def __init__(self, total_sentences=0):
		self.total_sentences = total_sentences
		
	def to_json(self):
		"""
		
		Returns a JSON representation of ``BatchStats``.
		
		:return: A JSON dictionary representation of the ``BatchStats``
		:rtype: ``dict``
		
		"""
		return { "total_sentences": self.total_sentences }