from ober.tokens import TokenDatabase
from ober.documents import DocumentDatabase

from collections import defaultdict
import os
import operator
import argparse

def count_vocab(document_database):
	counts = defaultdict(int)
	for sentence in document_database.get_sentences():
		for token in sentence:
			counts[token] += 1
	return dict(counts)

def main():
	parser = argparse.ArgumentParser(description="Add new tokens, update counts, and add new token vectors from a new document inventory")
	parser.add_argument("--token_db_path", help="path to token inventory", default="data/tokens")
	parser.add_argument("--document_db_path", help="path to document inventory", default="data/documents")
	parser.add_argument("--document_set", help="document set to update from", default="trigrams")
	parser.add_argument("--min_count", help="minimum number of times a token must appear to be added", type=int, default=5)
	args = parser.parse_args()
	
	# load document database
	document_database = DocumentDatabase.load(db_path=args.document_db_path, document_set=args.document_set)
	
	# count vocab
	print("Counting vocab ...")
	counts = count_vocab(document_database)
	
	# sort vocab and remove words of less than the min count
	print("Sorting vocab ...")
	counts = { token: counts[token] for token in counts if counts[token] >= args.min_count }
	counts = sorted(counts.items(), key=operator.itemgetter(1))
	counts.reverse()
	
	# load old token database and vectors
	print("Loading old database ...")
	old_token_database = TokenDatabase.load(db_path=args.token_db_path)
	old_vectors = old_token_database.get_vectors()
	
	# create new TokenDatabase
	print("Creating new database ...")
	new_token_database = TokenDatabase(vector_size=old_token_database.vector_size)
	
	# add all vocabulary
	print("Adding vocabulary ...")
	for token in counts:
		new_token_database.add_token(token[0], count=token[1])
		
	# create new vectors
	print("Creating new vectors ...")
	new_token_database.generate_random_vectors()
	new_vectors = new_token_database.get_vectors()
	
	# copy over and existing vectors from previous version
	print("Copying exisiting vectors ...")
	for token, _ in counts:
		if token in old_token_database:
			new_vectors[new_token_database.encode_token(token)] = old_vectors[old_token_database.encode_token(token)]
			
	print("Saving ...")		
	
	# update vectors
	new_token_database.update_vectors(new_vectors)
	
	# save
	new_token_database.save(db_path=args.token_db_path, version=old_token_database.version + 1, vectors_version=1)
	
if __name__ == "__main__":
	main()