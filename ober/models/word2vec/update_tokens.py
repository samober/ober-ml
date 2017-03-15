import operator
import argparse

from collections import defaultdict

from ober.tokens import TokenDatabase
from ober.documents import DocumentDatabase

def count_vocab(document_database):
	
	"""
	
	Counts the number of times each token occurs in the text data.
	
	"""
	
	# create empty dictionary
	counts = defaultdict(int)
	# loop through every sentence
	for sentence in document_database.get_sentences():
		# loop through every token
		for token in sentence:
			# increment the count by one each time
			counts[token] += 1
			
	# cast to a immutable dictionary and return
	return dict(counts)

def main():
	
	"""
	
	This script examines new additions to the document databases, updates the token counts and adds new tokens, and generates new token vectors.
	
	Once new vectors are created, any tokens that were present in a previous version of the database will have their previous token vector transferred to the current database.
	
	"""
	
	# create an argument parser
	parser = argparse.ArgumentParser(description="Add new tokens, update counts, and add new token vectors from a new document inventory")
	
	# add arguments
	parser.add_argument("--tokens_path", help="The path to token directory.", default="data/tokens")
	parser.add_argument("--documents_path", help="The path to document directory.", default="data/documents")
	parser.add_argument("--documents_version", help="The version of the documents database to load.", type=int)
	parser.add_argument("--min_count", help="The minimum number of times a token must appear to be added to the new database.", type=int, default=5)
	
	# parse the arguments
	args = parser.parse_args()
	
	# resolve the documents version
	documents_version = args.documents_version
	if not documents_version:
		documents_version = DocumentDatabase.get_latest_version(args.documents_path)
	
	# load document database
	document_database = DocumentDatabase.load(args.documents_path, documents_version)
	
	# print setup information
	print ""
	print "OBER - TOKEN FREQUENCY AND VECTOR GENERATION SCRIPT"
	print ""
	print ""
	print "OLD TOKENS:\t\t%s" % args.tokens_path
	print "DOCUMENTS:\t\t%s [VERSION: %d]" % (args.documents_path, documents_version)
	print "MINIMUM COUNT ALLOWED:\t\t%d" % args.min_count
	print ""
	print ""
	
	# count vocab
	print("COUNTING VOCAB ...")
	counts = count_vocab(document_database)
	
	# sort vocab and remove words of less than the min count
	print("SORTING VOCAB ...")
	# filter by count
	counts = { token: counts[token] for token in counts if counts[token] >= args.min_count }
	# sort by count ascending
	counts = sorted(counts.items(), key=operator.itemgetter(1))
	# reverse to get descending
	counts.reverse()
	
	# load old token database and vectors
	print("LOADING OLD TOKEN DATABASE ...")
	old_token_database = TokenDatabase.load(db_path=args.tokens_path)
	# save the old vectors
	old_vectors = old_token_database.get_vectors()
	
	# create new TokenDatabase with same vector size and increment the version
	print("CREATING NEW TOKEN DATABASE ...")
	new_token_database = TokenDatabase(vector_size=old_token_database.vector_size, version=old_token_database.version + 1)
	
	# add all vocabulary
	print("TRANSFERING VOCABULARY ...")
	# loop through each token and add to the new database
	for token in counts:
		new_token_database.add_token(token[0], count=token[1])
		
	# create new vectors
	print("GENERATING NEW VECTORS ...")
	new_token_database.generate_random_vectors()
	# save the new vectors
	new_vectors = new_token_database.get_vectors()
	
	# copy over any existing vectors from previous version
	print("TRANSFERING EXISTING VECTORS ...")
	# loop through each token in the new database
	for token, _ in counts:
		# check if it is in the old database as well
		if token in old_token_database:
			# if it is, copy over the token vector using the token ids
			new_vectors[new_token_database.encode_token(token)] = old_vectors[old_token_database.encode_token(token)]
			
	print("SAVING ...")		
	
	# update vectors
	new_token_database.update_vectors(new_vectors)
	
	# save (set new flags to false because we have already set the correct versions before)
	new_token_database.save(new_version=False, new_vectors_version=False)
	
if __name__ == "__main__":
	main()