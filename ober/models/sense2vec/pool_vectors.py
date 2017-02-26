from ober.tokens import TokenDatabase
from ober.senses import SenseDatabase

import struct
import numpy as np
import argparse
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
	
def read_clusters(clusters_file):
	# sense struct (token id, sense id, num neighbors)
	sense_struct = struct.Struct(">i i i")
	# cluster member struct (neighbor id, weight)
	member_struct = struct.Struct(">i f")
	with open(clusters_file, "rb") as f:
		while True:
			data = f.read(12)
			if not data:
				break
			token_id, sense_id, num_members = sense_struct.unpack(data)
			
			# read in members
			members = []
			for _ in range(num_members):
				members.append(member_struct.unpack(f.read(8)))
			
			# return
			yield (token_id, sense_id, members)

def main():
	parser = argparse.ArgumentParser(description="Create sense vectors out of a clusters file and word vectors")
	parser.add_argument("--token_db_path", help="path to the token inventory", default="data/tokens")
	parser.add_argument("--token_db_version", help="version of the token inventory to use", type=int)
	parser.add_argument("--token_vectors_version", help="version of the token vectors to use", type=int)
	parser.add_argument("--sense_db_path", help="path to the sense inventory", default="data/senses")
	parser.add_argument("--sense_db_version", help="version of the sense inventory to write to", type=int)
	parser.add_argument("--sense_vectors_version", help="version of the sense vectors to write to", type=int)
	parser.add_argument("--clusters_path", help="path to the clusters inventory", default="data/senses/clusters")
	parser.add_argument("--clusters_version", help="version of the clusters inventory to use", type=int)
	args = parser.parse_args()
	
	token_db_version = args.token_db_version
	if not token_db_version:
		token_db_version = TokenDatabase.get_latest_version(args.token_db_path)
	token_vectors_version = args.token_vectors_version
	if not token_vectors_version:
		token_vectors_version = TokenDatabase.get_latest_vectors_version(args.token_db_path, token_db_version)
		
	sense_db_version = args.sense_db_version
	if not sense_db_version:
		sense_db_version = SenseDatabase.get_latest_version(args.sense_db_path) + 1
	sense_vectors_version = args.sense_vectors_version
	if not sense_vectors_version:
		sense_vectors_version = SenseDatabase.get_latest_vectors_version(args.sense_db_path, sense_db_version) + 1
		
	clusters_version = args.clusters_version
	if not clusters_version:
		clusters_version = get_latest_clusters_version(args.clusters_path)
	
	print ("Loading databases ...")
	# load token database
	token_database = TokenDatabase.load(db_path=args.token_db_path, version=token_db_version, vectors_version=token_vectors_version)
	# create sense database
	sense_database = SenseDatabase(db_path=args.sense_db_path, version=sense_db_version, vectors_version=sense_vectors_version)
	
	print("Loading/creating vectors ...")
	# load old vectors
	token_vectors = token_database.get_vectors()
	# create new vectors
	sense_vectors = np.zeros((200000, sense_database.vector_size), dtype=np.float32)
	
	print ("Pooling vectors ...")
	# load clusters			
	for index, cluster in enumerate(read_clusters(os.path.join(args.clusters_path, "%05d.clusters" % clusters_version))):
		sense_database.add_sense("%s#%d" % (token_database.decode_token(cluster[0]), cluster[1]))
		# make sure there is enough room in sense vectors
		while index >= sense_vectors.shape[0]:
			sense_vectors = np.concatenate((sense_vectors, np.zeros_like(sense_vectors)), axis=0)
		sense_vector = sense_vectors[index]
		weight_total = 0
		for member in cluster[2]:
			sense_vector += token_vectors[member[0]] * member[1]
			weight_total += member[1]
		sense_vector /= weight_total
		sense_vectors[index] = sense_vector
		
	print ("Saving ...")
	# update vectors and save
	sense_database.update_vectors(sense_vectors[:len(sense_database)])
	sense_database.save()
	
if __name__ == "__main__":
	main()