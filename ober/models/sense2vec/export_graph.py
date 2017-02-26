from ober.tokens import TokenDatabase

import argparse

def main():
	parser = argparse.ArgumentParser(description="exports the token similarity graph for a token inventory")
	parser.add_argument("--token_db_path", help="path to token inventory", default="data/tokens")
	parser.add_argument("--tokens_version", help="version of token inventory to export", type=int)
	parser.add_argument("--vectors_version", help="version of vectors to export", type=int)
	parser.add_argument("--graph_version", help="version of graph to export", type=int)
	args = parser.parse_args()
	
	tokens_version = args.tokens_version
	if not tokens_version:
		tokens_version = TokenDatabase.get_latest_version(args.token_db_path)
	vectors_version = args.vectors_version
	if not vectors_version:
		vectors_version = TokenDatabase.get_latest_vectors_version(args.token_db_path, tokens_version)
	graph_version = args.graph_version
	if not graph_version:
		graph_version = TokenDatabase.get_latest_graph_version(db_path=args.token_db_path, version=tokens_version) + 1
	
	# load token database
	token_database = TokenDatabase.load(db_path=args.token_db_path, version=tokens_version, vectors_version=vectors_version)
	
	# export
	print "Exporting graph version %d ..." % graph_version
	token_database.export_distributed_graph(graph_version=graph_version)
	
if __name__ == "__main__":
	main()