import os
import bz2
import json
import codecs
import random

from ober.data import VersionedFile

class DocumentDatabase:
	"""
	
	A ``DocumentDatabase`` manages a set of documents in the document inventory with a specified version.
	
	For example, you could have a ``DocumentDatabase`` of all documents in a directory called "data/documents/bigrams", which store documents after being run through a bigram phraser. 
	
	You could also have multiple versions of your inventory. For example, if you add a sizeable number of new documents and update your phraser, you could then rephrase all your documents into version 2 of the bigrams document set.
	
	This is illustrated below:
		
	.. code-block:: python
	
		from ober.documents import DocumentDatabase
		
		# load version 1 of a document database located at 'data/documents/bigrams' 
		document_database = DocumentDatabase.load(db_path="data/documents/bigrams", version=1)
		
		### DO SOME WORK
		
		# save the new documents to version 2
		document_database.save("data/documents/bigrams", version=2)
		
	Documents are stored as JSON lines files by batches:
		
	.. code-block:: python
		
		<inventory_directory>
			<version_id>
				<batch_id>
					data.jl.bz2
					stats.json
				<batch_id>
					data.jl.bz2
					stats.json
				[...]
			<version_id>
				<batch_id>
					data.jl.bz2
					stats.json
				<batch_id>
					data.jl.bz2
					stats.json
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
	
	def __init__(self, db_path="data/documents/trigrams", version=None):
		self.db_path = db_path
		self.version = version
		
		# create a hash to batch statistics
		self.batch_stats = {}
		
		# create the base versioned file system
		self.file_base = VersionedFile(db_path)
		# if there is no version specified pick the latest one
		if not self.version:
			self.version = self.file_base.get_latest_version()
		# if there are no versions create a new one
		if self.version == 0:
			self.version = self.file_base.create_latest_version()
		
		# create batches file system under current version
		self.version_batches = VersionedFile(self.file_base.get_version_path(self.version), version_num_length=4)
		
	def _get_batches(self):
		return self.version_batches.get_versions()
		
	def _get_batch_files(self):
		return [ batch_file[1] for batch_file in self.version_batches.get_file_paths("data.jl.bz2") ]
		
	def _get_batch_file(self, batch=1):
		return self.version_batches.get_file_path(batch, "data.jl.bz2")
		
	def _get_batch_stat_file(self, batch=1):
		return self.version_batches.get_file_path(batch, "stats.json")
			
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
		more_documents = True
		while more_documents:
			# increment batch number
			batch = len(self.batch_stats) + 1
			sentences_count = 0
			# create temporary batch data file
			batch_file = os.path.join(self.file_base.path, "data.jl.bz2.temp")
			# try to read the next batch of files, catch exception and stop if there are no more
			try:
				document = documents.next()
				with bz2.BZ2File(batch_file, "w") as outfile:
					for i in range(500000):
						# count sentences
						for paragraph in document["paragraphs"]:
							sentences_count += len(paragraph["sentences"])
						# write JSON to file
						outfile.write("%s\n" % json.dumps(document))
						# if we are not done with this batch, retrieve the next document
						if i != 499999:
							document = documents.next()
			except StopIteration:
				# the end of the documents stream
				more_documents = False
			if sentences_count > 0:
				# create the new batch in the file system
				self.version_batches.create_latest_version()
				# write the batch statistics to file and save
				batch_stat_file = self._get_batch_stat_file(batch)
				self.batch_stats[batch] = BatchStats(sentences_count)
				with codecs.open(batch_stat_file, "wb", "utf-8") as outfile:
					outfile.write(json.dumps(self.batch_stats[batch].to_json()))
				# move temp data file to the correct location
				os.rename(batch_file, self._get_batch_file(batch)) 
	
	@staticmethod
	def load(db_path="data/documents/trigrams", version=None):
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
		# create database at the desired path and with the desired version
		db = DocumentDatabase(db_path, version)
		# loop through batches
		for batch in db._get_batches():
			# get the path to the stats file
			stats_file = db._get_batch_stat_file(batch)
			# load the stats
			stats_json = json.loads(codecs.open(stats_file, "rb", "utf-8").read())
			# save in the batch statistics hash
			db.batch_stats[batch] = BatchStats(stats_json["total_sentences"])
		return db
		
	@staticmethod
	def get_latest_version(db_path):
		"""
		
		Returns the latest version of the documents inventory at the specified path.
		
		:param db_path: The path to the documents file system.
		:type db_path: str
		:return: The version id for the latest version.
		:rtype: int
		
		"""
		# create a file system
		file_base = VersionedFile(db_path)
		return file_base.get_latest_version()
		
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