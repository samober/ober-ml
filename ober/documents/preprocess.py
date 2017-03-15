import spacy
import itertools
import sys
import argparse

from textacy.corpora.reddit_reader import RedditReader

from ober.documents import DocumentDatabase

def split_into_paragraphs(documents):
	"""
	
	Takes reddit corpus JSON documents and creates an iterator of tuples in the format (document index, document name, paragraph text).
	
	"""
	for index, document in enumerate(documents):
		for paragraph in document["body"].split("\n"):
			text = paragraph.strip()
			if len(text) > 0:
				yield (index, document["name"], text)

def combine_noun_chunks(nlp, texts):
	"""
	
	Takes an iterator of texts and a spacy parser and creates a high throughput generator of JSON paragraph representations.
	
	Merges all noun chunks into single tokens, lowercases all text and strips extra whitespace, and then returns a Python dictionary for the paragraph.
	
	"""
	for doc in nlp.pipe(texts, n_threads=4, batch_size=2000):
		# condense nouns
		for np in doc.noun_chunks:
			while len(np) > 1 and np[0].dep_ in [ "det", "nummod", "poss" ]:
				np = np[1:]
			if len(np) > 1:
				np.merge(np.root.tag_, np.text, np.root.ent_type_)
				
		# create sentences array
		sentences = []
		
		# loop through sentences
		for sentence in doc.sents:
			# create tokens array
			tokens = []
			for token in sentence:
				tokens.append(token.lower_.strip().replace(" ", "_"))
			sentences.append({ "tokens": tokens })
			
		yield { "sentences": sentences }

def main():
	
	"""
	
	This script takes the reddit corpus archive dump and parses it into a tokenized JSON representation for later use in a Word2Vec model by merging noun phrases, separating punctuation, and lowercasing all text.
	
	It uses the DocumentDatabase class to stream these JSON documents to a compressed archive on disk.
	
	"""
	
	# create argument parser
	parser = argparse.ArgumentParser(description="Pre-processor for reddit corpus dumps. Parses text and stores it in a DocumentDatabase inventory.")
	
	# add arguments
	parser.add_argument("--documents_path", help="The path to the documents directory.", default="data/documents/noun_chunked")
	parser.add_argument("--documents_version", help="The version of the document database to save to. (Defaults to a new version.)", type=int)
	parser.add_argument("--reddit_path", help="The path to the reddit corpus archive.", default="data/raw/reddit/reddit_corpus.gz")
	
	# parse
	args = parser.parse_args()
	
	# resolve documents version
	documents_version = args.documents_version
	if not documents_version:
		documents_version = DocumentDatabase.get_latest_version(args.documents_path) + 1
		
	# print setup information
	print ""
	print "OBER TEXT PREPROCESSOR (NOUN CHUNK - REDDIT CORPUS DUMP)"
	print ""
	print "REDDIT ARCHIVE:\t\t%s" % args.reddit_path
	print "SAVING TO:\t\t%s [VERSION: %d]" % (args.documents_path, documents_version)
	print ""
	print ""
	
	# load spacy
	print "LOADING SPACY NLP LIBRARY ..."
	nlp = spacy.load("en")

	# load the reddit reader
	print "LOADING TEXTACY REDDIT CORPUS READER ..."
	reader = RedditReader(args.reddit_path)
	
	# load the document database
	print "LOADING DOCUMENT DATABASE ..."
	document_database = DocumentDatabase.load(args.documents_path, version=documents_version)
	
	# get iterator of documents
	documents = reader.records(min_len=200)
	
	# split documents into paragraphs (document id, document title, paragraph)
	paragraphs = split_into_paragraphs(documents)
	
	# split iterator into two
	paragraphs_1, paragraphs_2 = itertools.tee(paragraphs)
	
	# one keeps index and titles
	paragraphs_1 = ( (paragraph[0], paragraph[1]) for paragraph in paragraphs_1 )
	# the other just keeps text
	paragraphs_2 = ( paragraph[2] for paragraph in paragraphs_2 )
	
	# combine noun chunks for the texts
	paragraphs_2 = combine_noun_chunks(nlp, paragraphs_2)
	
	# zip paragraphs_1 and paragraphs_2 back together
	paragraphs = itertools.izip(paragraphs_1, paragraphs_2)
	
	# group by index
	documents = itertools.groupby(paragraphs, lambda x: x[0])
	
	# format into JSON objects
	documents = ( { "title": document[0][1], "paragraphs": [ paragraph[1] for paragraph in document[1] ] } for document in documents )
	
	# begin parsing
	print "\nBEGINNING PARSE ..."
	document_database.add_documents(documents)
	
if __name__ == "__main__":
	main()