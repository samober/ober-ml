from ober.tokens import TokenDatabase

import argparse
import subprocess
import sys
import os

def get_latest_clusters_version(clusters_path="data/senses/clusters"):
	max_version = 0
	for f in os.listdir(clusters_path):
		if f.endswith(".clusters"):
			version = 0
			try:
				version = int(f.split(".")[0])
			except:
				version = 0
			if version > max_version:
				max_version = version
	return max_version

def main():
	parser = argparse.ArgumentParser(description="Clustering algorithm for token similarity graph")
	parser.add_argument("--tokens_db_path", help="location of the token inventory store (defaults to data/tokens)", default="data/tokens")
	parser.add_argument("--tokens_version", help="version of the token inventory to use (defaults to latest)", type=int)
	parser.add_argument("--graph_version", help="version of the similarity graph to use (defaults to latest", type=int)
	parser.add_argument("--clusters_path", help="path to the clusters inventory", default="data/senses/clusters")
	parser.add_argument("--clusters_version", help="version of the clusters to output", type=int)
	parser.add_argument("--num_workers", help="number of worker threads", type=int, default=4)
	args = parser.parse_args()
	
	tokens_db_path = args.tokens_db_path
	
	tokens_version = args.tokens_version
	if not tokens_version:
		tokens_version = TokenDatabase.get_latest_version(db_path=tokens_db_path)
		
	graph_version = args.graph_version
	if not graph_version:
		graph_version = TokenDatabase.get_latest_graph_version(db_path=tokens_db_path, version=tokens_version)
	
	graph_file = TokenDatabase.get_graph_path(db_path=tokens_db_path, version=tokens_version, graph_version=graph_version)
	
	clusters_version = args.clusters_version
	if not clusters_version:
		clusters_version = get_latest_clusters_version(args.clusters_path) + 1
	clusters_file = os.path.join(args.clusters_path, "%05d.clusters" % clusters_version)
	
	bash_command = "java -jar -Xms4G -Xmx16G ober/models/sense2vec/chinese-whispers/target/chinese-whispers-1.0.0.jar --graph %s --output %s --num_workers %d" % (graph_file, clusters_file, args.num_workers)
	
	process = subprocess.Popen(bash_command.split(), stdout=subprocess.PIPE)
	
	for line in iter(process.stdout.readline, ''):
		sys.stdout.write(line)

if __name__ == "__main__":
	main()