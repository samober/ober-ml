package com.ober.ml.cw;

import com.ober.ml.cw.graph.Graph;
import com.ober.ml.cw.graph.Edge;
import com.ober.ml.cw.utils.LEDataInputStream;

import java.io.FileInputStream;
import java.io.DataInputStream;
import java.io.FileOutputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.io.EOFException;
import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;

import java.util.List;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Set;

import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.ArrayBlockingQueue;

public class WSI {
	
	private Graph distributedGraph;
	
	private int maxEdges;
	private int maxConnectivity;
	private int maxIterations;
	private int minCluster;
	private int numWorkers;
	
	private String infile;
	private String outfile;
	
	public WSI(String infile, String outfile) {
		this(infile, outfile, 200, 200, 100, 5, 4);
	}
	
	public WSI(String infile, String outfile, int maxEdges, int maxConnectivity, int maxIterations, int minCluster, int numWorkers) {
		this.infile = infile;
		this.outfile = outfile;
		this.maxEdges = maxEdges;
		this.maxConnectivity = maxConnectivity;
		this.maxIterations = maxIterations;
		this.minCluster = minCluster;
		this.numWorkers = numWorkers;
	}
	
	public void loadGraph() {
		System.out.println("\n\n### CHINESE WHISPERS ###\n");
		
		System.out.println("MAX EDGES: " + this.maxEdges);
		System.out.println("MAX CONNECTIVITY: " + this.maxConnectivity);
		System.out.println("MAX ITERATIONS: " + this.maxIterations);
		System.out.println("MIN CLUSTER SIZE: " + this.minCluster);
		System.out.println("NUM WORKERS: " + this.numWorkers);
		System.out.println("\n#########################\n");
		
		System.out.println("LOADING: " + this.infile);
		long startTime = System.currentTimeMillis();
		this.distributedGraph = this.loadDistributedGraph(this.infile);
		float totalTime = (System.currentTimeMillis() - startTime) / 1000.0f;
		System.out.println("LOAD: " + totalTime + " seconds");
		System.out.println("TOTAL NODES: " + this.distributedGraph.getSize() + "\n");
		
		System.out.println("SORTING GRAPH EDGES");
		startTime = System.currentTimeMillis();
		this.distributedGraph.sortEdges();
		totalTime = (System.currentTimeMillis() - startTime) / 1000.0f;
		System.out.println("SORT: " + totalTime + " seconds\n");
	}
	
	public void calculateSenses() {
		// create workers list
		ArrayList<Thread> workers = new ArrayList<Thread>(numWorkers);
		// calculate batch size
		int batchSize = distributedGraph.getSize() / numWorkers;
			
		// create queue for finished clusters
		BlockingQueue<Cluster> clusterQueue = new ArrayBlockingQueue<Cluster>(1024);
			
		// create progress counter
		AtomicInteger progress = new AtomicInteger(0);
		AtomicInteger totalClusters = new AtomicInteger(0);
		
		// start workers
		for (int i = 0; i < this.numWorkers; i++) {
			int batchStart = i * batchSize;
			int batchEnd = i == numWorkers - 1 ? this.distributedGraph.getSize() : (i + 1) * batchSize;
			Thread worker = new Thread(new Worker(this.distributedGraph, clusterQueue, progress, batchStart, batchEnd, maxEdges, maxConnectivity, maxIterations, minCluster));
			worker.setName("WORKER-" + (i+1));
			worker.start();
			workers.add(worker);
		}
		
		// start cluster writer
		Thread writerThread = new Thread(new ClusterWriter(this.outfile, clusterQueue, totalClusters));
		writerThread.start();
		
		long startTime = System.currentTimeMillis();
		
		// print progress
		while (progress.get() < this.distributedGraph.getSize()) {
			int p = progress.get();
			if (p > 0 && p % 10 == 0) {
				System.out.println("PROGRESS: " + p + " nodes completed");
			}
			while (progress.get() == p) { }
		}
		
		// join all workers
		for (Thread worker : workers) {
			try {
				worker.join();
			} catch (InterruptedException e) {
				e.printStackTrace();
			}
		}
		
		// add cluster with node -1 to queue to signal the end
		try {
			clusterQueue.put(new Cluster(-1, -1));
		} catch (InterruptedException e) {
			e.printStackTrace();
		}
		
		// join the cluster writer
		try {
			writerThread.join();
		} catch (InterruptedException e) {
			e.printStackTrace();
		}
		
		// report time
		float totalTime = (System.currentTimeMillis() - startTime) / 1000.0f;
		System.out.println("\n#########################\n");
		System.out.println("TOTAL CLUSTERS: " + totalClusters.get());
		System.out.println("CALCULATE CLUSTERS: " + totalTime + " seconds\n");
	}
	
	private class Worker implements Runnable {
		
		private Graph distributedGraph;
		private BlockingQueue<Cluster> clusterQueue;
		private AtomicInteger progress;
		private int batchStart;
		private int batchEnd;
		
		private int maxEdges;
		private int maxConnectivity;
		private int maxIterations;
		private int minCluster;
		
		public Worker(Graph distributedGraph, BlockingQueue<Cluster> clusterQueue, AtomicInteger progress, int batchStart, int batchEnd, int maxEdges, int maxConnectivity, int maxIterations, int minCluster) {
			this.distributedGraph = distributedGraph;
			this.clusterQueue = clusterQueue;
			this.progress = progress;
			this.batchStart = batchStart;
			this.batchEnd = batchEnd;
			this.maxEdges = maxEdges;
			this.maxConnectivity = maxConnectivity;
			this.maxIterations = maxIterations;
			this.minCluster = minCluster;
		}
		
		@Override
		public void run() {
			System.out.println(Thread.currentThread().getName() + ": STARTING BATCH " + this.batchStart + " -> " + this.batchEnd);
			
			// loop through batch
			for (int i = this.batchStart; i < this.batchEnd; i++) {
				Graph graph = createTokenGraph(i);
				chineseWhispers(graph);
				findClusters(i, graph);
				this.progress.incrementAndGet();
			}
		}
		
		private Graph createTokenGraph(int node) {
			// create the graph
			Graph graph = new Graph();
		
			// get the closest neighbors to the token
			List<Integer> neighbors = this.distributedGraph.getNeighbors(node);
			Set<Integer> neighborsSet = new HashSet<Integer>(neighbors); // speed boost later
		
			// loop through the neighbors
			for (int i = 0; i < Math.min(this.maxEdges, neighbors.size()); i++) {
				Integer neighbor = neighbors.get(i);
				
				// loop through the closest neighbors to this neighbor
				List<Edge<Integer, Float>> foreignNeighbors = this.distributedGraph.getEdges(neighbor);
				for (int j = 0; j < Math.min(this.maxConnectivity, foreignNeighbors.size()); j++) {
					Integer foreignNeighbor = foreignNeighbors.get(j).getNode();
					Float weight = foreignNeighbors.get(j).getWeight();
					// if foreign neighbor is neighbors with the target node
					// add a new auxiliary edge to the target node's ego network
					if (foreignNeighbor != node && neighborsSet.contains(foreignNeighbor)) {
						graph.addEdge(neighbor, foreignNeighbor, weight);
					}
				}			
			}
		
			return graph;
		}
		
		private void chineseWhispers(Graph tokenNetwork) {
			// initialize random classes
			List<Integer> n = tokenNetwork.getNodes();
			for (int i = 1; i < n.size()+1; i++)
				tokenNetwork.setNodeClass(n.get(i-1), i);
		
			// run chinese whispers
			boolean changed = true;
			for (int z = 0; z < this.maxIterations; z++) {
				if (!changed)
					break;
				changed = false;
			
				// get all nodes and shuffle them
				List<Integer> nodes = tokenNetwork.getNodes();
				Collections.shuffle(nodes);
				for (Integer node : nodes) {
					List<Edge<Integer, Float>> neighbors = tokenNetwork.getEdges(node);
					HashMap<Integer, Float> classes = new HashMap<Integer, Float>();
					// collect all neighbors and sum edge weights per class
					for (Edge<Integer, Float> neighbor : neighbors) {
						Integer cls = tokenNetwork.getNodeClass(neighbor.getNode());
						Float weight = neighbor.getWeight();
					
						if (classes.containsKey(cls)) {
							classes.put(cls, classes.get(cls) + weight);
						} else {
							classes.put(cls, weight);
						}
					}
					// find the class with highest weight sum
					float max = -10000f;
					Integer maxClass = 0;
					for (Integer c : classes.keySet()) {
						if (classes.get(c) > max) {
							max = classes.get(c);
							maxClass = c;
						}
					}
					// set the class of the current node to the winning class and mark a change
					if (tokenNetwork.getNodeClass(node) != maxClass) {
						tokenNetwork.setNodeClass(node, maxClass);
						changed = true;
					}
				}
			}
		}
		
		private void findClusters(Integer baseNode, Graph tokenGraph) {
			// get a copy of all the nodes
			Set<Integer> nodes = new HashSet<Integer>(tokenGraph.getNodes());
			// temp list to track which nodes to remove
			Set<Integer> toRemove = new HashSet<Integer>();
			// current class value for the cluster
			Integer currentClass = 0;
			// current sense id
			Integer sense = 0;
			
			// map nodes to edge weights for fast lookup
			List<Edge<Integer, Float>> edges = this.distributedGraph.getEdges(baseNode);
			HashMap<Integer, Float> weights = new HashMap<Integer, Float>(edges.size());
			for (Edge<Integer, Float> edge : edges)
				weights.put(edge.getNode(), edge.getWeight());
			
			// loop through until no more nodes left
			while (nodes.size() > 0) {
				// create a new cluster
				Cluster cluster = new Cluster(baseNode, sense);
				// loop through all nodes and add them to the cluster if their class matches
				for (Integer node : nodes) {
					if (currentClass == 0)
						currentClass = tokenGraph.getNodeClass(node);
					if (tokenGraph.getNodeClass(node) == currentClass) {
						cluster.addMember(node, weights.get(node));
						// make sure to remove
						toRemove.add(node);
					}
				}
				// remove all added nodes
				nodes.removeAll(toRemove);
				toRemove.clear();
				// reset current class to zero so it gets updated again
				currentClass = 0;
				// if this cluster meets minimum size
				if (cluster.getMembers().size() >= this.minCluster) {
					// increment sense id
					sense++;
					// add the new cluster to the queue
					try {
						this.clusterQueue.put(cluster);
					} catch (InterruptedException e) {
						e.printStackTrace();
					}
				}
			}
		}
		
	}
	
	private class ClusterWriter implements Runnable {
		
		private String outfile;
		private BlockingQueue<Cluster> clusterQueue;
		private AtomicInteger totalClusters;
		
		public ClusterWriter(String outfile, BlockingQueue<Cluster> clusterQueue, AtomicInteger totalClusters) {
			this.outfile = outfile;
			this.clusterQueue = clusterQueue;
			this.totalClusters = totalClusters;
		}
		
		@Override
		public void run() {
			try {
				DataOutputStream outputStream = new DataOutputStream(new BufferedOutputStream(new FileOutputStream(this.outfile)));
				
				while (true) {
					try {
						Cluster cluster = this.clusterQueue.take();
						// check if we receive final cluster (has node of -1)
						if (cluster.getNode() == -1) {
							// we are finished
							break;
						}
						writeCluster(cluster, outputStream);
						totalClusters.incrementAndGet();
					} catch (InterruptedException e) {
						e.printStackTrace();
					}
				}
				
				outputStream.close();
			} catch (IOException e) {
				e.printStackTrace();
			}
		}
		
		private void writeCluster(Cluster cluster, DataOutputStream out) throws IOException {
			// write node id
			out.writeInt(cluster.getNode());
			// write sense id
			out.writeInt(cluster.getSense());
			// write number of members
			out.writeInt(cluster.getMembers().size());
			// write members
			for (ClusterMember member : cluster.getMembers()) {
				// write member node id
				out.writeInt(member.getNode());
				// write member weight
				out.writeFloat(member.getWeight());
			}
		}
		
	}
	
	public static Graph loadDistributedGraph(String filename) {
		// create the graph
		Graph graph = new Graph(200000, 220);
		
		// read in edges
		try {
			LEDataInputStream inputStream = new LEDataInputStream(new BufferedInputStream(new FileInputStream(filename)));
			while (inputStream.available() > 0) {
//			for (int i = 0; i < 100000; i++) {
//				System.out.println(inputStream.readInt() + ", " + inputStream.readInt() + ", " + inputStream.readFloat());
				graph.addEdge(inputStream.readInt(), inputStream.readInt(), inputStream.readFloat());
			}
			inputStream.close();
		} catch (IOException e) {
			e.printStackTrace(System.out);
			System.exit(1);
		}
		
		return graph;
	}
	
}