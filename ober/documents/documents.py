import os
import gzip
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
		
	Documents are stored as JSON lines files by batches (gzip compressed):
		
	.. code-block:: python
		
		<inventory_directory>
			<version_id>
				<batch_id>
					data.jl.gz
					stats.json
				<batch_id>
					data.jl.gz
					stats.json
				[...]
			<version_id>
				<batch_id>
					data.jl.gz
					stats.json
				<batch_id>
					data.jl.gz
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
	
	# batch size for stream-adding files to the database
	DOCUMENT_BATCH_SIZE = 500000
	
	def __init__(self, db_path="data/documents/trigrams", version=None):
		# initialize db_path and version
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
		
	# HELPER FUNCTIONS
	
	# gets the integer list of all batches
	def _get_batches(self):
		return self.version_batches.get_versions()
		
	# gets the string list of all the direct paths to the batch data files
	def _get_batch_files(self):
		return [ batch_file[1] for batch_file in self.version_batches.get_file_paths("data.jl.gz") ]
		
	# gets a specific path for a batch
	def _get_batch_file(self, batch=1):
		return self.version_batches.get_file_path(batch, "data.jl.gz")
		
	# gets a specific path for a batch statistics file
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
			# no batch = all the batches
			files = self._get_batch_files()
		elif batch == "random":
			# get all the batches and pick one from random
			batches = self._get_batches()
			files = [ self._get_batch_file(batch=random.randint(1, len(batches))) ]
		else:
			# get the specified batch
			files = [ self._get_batch_file(batch=batch) ]
			
		# loop through all the batch files
		for f in files:
			with gzip.open(f, "rb") as infile:
				for line in infile:
					# parse the JSON for each line
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
		
		# loop through the document stream for this document database
		for document in self.get_documents(batch):
			for paragraph in document["paragraphs"]:
				# yield the paragraphs one by one
				yield paragraph
		
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
		
		# loop through the paragraph stream for this document database
		for paragraph in self.get_paragraphs(batch):
			# loop through the sentences
			for sentence in paragraph["sentences"]:
				# yield the individual tokens
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
		
		# loop through batches and add up all their individual sentence counts
		total_sentences = 0
		for batch in self.batch_stats:
			total_sentences += self.batch_stats[batch].total_sentences
		return total_sentences
		
	def add_documents(self, documents):
		
		"""
		
		Adds ``documents`` to the document inventory, writing to disk in batches of 500,000.
		
		:param documents: An iterator of JSON documents.
		:type documents: ``Iterator[dict]``
		
		"""
		
		# flag for StopIteration exceptions
		more_documents = True
		# loop while there are still documents in the iterator
		while more_documents:
			# increment batch number
			batch = len(self.batch_stats) + 1
			# count sentences
			sentences_count = 0
			# create temporary batch data file in the version directory
			batch_file = os.path.join(self.file_base.get_version_path(self.version), "data.jl.gz.temp")
			# try to read the next batch of files, catch exception and stop if there are no more
			try:
				# get next document before opening the file just to make sure it's there
				document = documents.next()
				# open the data file
				with gzip.open(batch_file, "wb") as outfile:
					# loop through DOCUMENT_BATCH_SIZE documents
					for i in range(DocumentDatabase.DOCUMENT_BATCH_SIZE):
						# count sentences in document
						for paragraph in document["paragraphs"]:
							sentences_count += len(paragraph["sentences"])
						# write JSON to file one line at a time
						outfile.write("%s\n" % json.dumps(document))
						# if we are not done with this batch, retrieve the next document
						if i < DocumentDatabase.DOCUMENT_BATCH_SIZE - 1:
							document = documents.next()
			except StopIteration:
				# the end of the documents stream, set the flag to False
				more_documents = False
			# make sure the batch isn't empty
			if sentences_count > 0:
				# create the new batch in the file system
				self.version_batches.create_latest_version()
				# add the stats to the statistics hash
				self.batch_stats[batch] = BatchStats(sentences_count)
				# write the batch statistics to file
				with codecs.open(self._get_batch_stat_file(batch), "wb", "utf-8") as outfile:
					# write the JSON representation for the stats
					outfile.write(json.dumps(self.batch_stats[batch].to_json()))
				# move the temp data file to the correct location inside the version folder
				os.rename(batch_file, self._get_batch_file(batch)) 
	
	@staticmethod
	def load(db_path="data/documents/trigrams", version=None):
		
		"""
		
		Loads a document database with the specified version from the directory.
		
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
			
		# return the database
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
		
		# create a file system and return latest version
		return VersionedFile(db_path).get_latest_version()
		
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