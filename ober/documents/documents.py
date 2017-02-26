import os
import bz2
import json
import codecs
import random

class DocumentDatabase:
	
	def __init__(self, db_path="data/documents", version=None, document_set="trigrams"):
		self.db_path = db_path
		self.version = version
		self.document_set = document_set
		self.batch_stats = {}
		
	def get_batches_(self):
		return DocumentDatabase.get_batches(self.db_path, self.version, self.document_set)
		
	def get_batch_files_(self):
		return DocumentDatabase.get_batch_files(self.db_path, self.version, self.document_set)
		
	def get_batch_file_(self, batch=1):
		return DocumentDatabase.get_batch_file(self.db_path, self.version, self.document_set, batch)
		
	def get_batch_stat_file_(self, batch=1):
		return DocumentDatabase.get_batch_stat_file(self.db_path, self.version, self.document_set, batch)
			
	def get_documents(self, batch=None):
		files = None
		if not batch:
			files = self.get_batch_files_()
		elif batch == "random":
			batches = self.get_batches_()
			files = [ self.get_batch_file_(batch=random.randint(1, len(batches))) ]
		else:
			files = [ self.get_batch_file_(batch=batch) ]
		for f in files:
			with bz2.BZ2File(f, "r") as infile:
				for line in infile:
					yield json.loads(line)
					
	def get_paragraphs(self, batch=None):
		for document in self.get_documents(batch):
			for paragraph in document["paragraphs"]:
				yield paragraph["sentences"]
		
	def get_sentences(self, batch=None):
		for paragraph in self.get_paragraphs(batch):
			for sentence in paragraph:
				yield sentence["tokens"]
				
	def get_batch_stats(self, batch):
		return self.batch_stats[batch]
		
	def num_batches(self):
		return len(self.batch_stats)
		
	def get_total_sentences(self):
		total_sentences = 0
		for batch in self.batch_stats:
			total_sentences += self.batch_stats[batch].total_sentences
		return total_sentences
		
	def add_documents(self, documents):
		DocumentDatabase.ensure_documents_dir(self.db_path, self.version, self.document_set)
		more_documents = True
		while more_documents:
			batch = len(self.batch_stats) + 1
			sentences_count = 0
			batch_file = self.get_batch_file_(batch) + ".temp"
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
				batch_stat_file = self.get_batch_stat_file_(batch)
				self.batch_stats[batch] = BatchStats(sentences_count)
				with codecs.open(batch_stat_file, "wb", "utf-8") as outfile:
					outfile.write(json.dumps(self.batch_stats[batch].to_json()))
				os.rename(batch_file, batch_file[:-5]) # take away .temp so it become valid batch
	
	@staticmethod
	def load(db_path="data/documents", version=None, document_set="trigrams"):
		if not version:
			version = DocumentDatabase.get_latest_version(db_path, document_set)
		db = DocumentDatabase(db_path, version, document_set)
		for batch in db.get_batches_():
			stats_file = db.get_batch_stat_file_(batch)
			stats_json = json.loads(codecs.open(stats_file, "rb", "utf-8").read())
			db.batch_stats[batch] = BatchStats(stats_json["total_sentences"])
		return db
	
	@staticmethod
	def get_latest_version(db_path="data/documents", document_set="trigrams"):
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
		return os.path.join(db_path, document_set, "%05d" % version)
		
	@staticmethod
	def get_batches(db_path, version, document_set):
		documents_directory = DocumentDatabase.get_documents_directory(db_path, version, document_set)
		return [ int(f.split(".")[0]) for f in os.listdir(documents_directory) if f.endswith(".jl.bz2") ]
		
	@staticmethod
	def get_batch_file(db_path, version, document_set, batch):
		return os.path.join(db_path, document_set, "%05d" % version, "%04d.jl.bz2" % batch)
		
	@staticmethod
	def get_batch_stat_file(db_path, version, document_set, batch):
		return os.path.join(db_path, document_set, "%05d" % version, "%04d.stats.json" % batch)
		
	@staticmethod
	def get_batch_files(db_path, version, document_set):
		return [ DocumentDatabase.get_batch_file(db_path, version, document_set, batch) for batch in DocumentDatabase.get_batches(db_path, version, document_set) ]
		
	@staticmethod
	def ensure_documents_dir(db_path, version, document_set):
		documents_directory = DocumentDatabase.get_documents_directory(db_path, version, document_set)
		if not os.path.exists(documents_directory):
			os.makedirs(documents_directory)
		
class BatchStats:
	def __init__(self, total_sentences=0):
		self.total_sentences = total_sentences
		
	def to_json(self):
		return { "total_sentences": self.total_sentences }