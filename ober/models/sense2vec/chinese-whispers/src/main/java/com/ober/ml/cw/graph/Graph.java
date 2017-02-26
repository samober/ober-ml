package com.ober.ml.cw.graph;

import java.util.BitSet;
import java.util.ArrayList;
import java.util.List;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;

import it.unimi.dsi.fastutil.ints.IntOpenHashSet;
import java.util.*;

public class Graph {
	
	protected int size;
	BitSet nodes;
	Integer[] nodeClasses;
	IntOpenHashSet[] edgeSourceSet;
	ArrayList<Integer>[] edgeSources;
	ArrayList<Float>[] edgeWeights;
	protected int initialNumEdgesPerNode; 
	
	public Graph() {
		this(10, 10);
	}
	
	public Graph(int initialSize) {
		this(initialSize, 10);
	}
	
	public Graph(int initialSize, int initialNumEdgesPerNode) {
		this.size = initialSize;
		nodes = new BitSet(initialSize);
		nodeClasses = new Integer[initialSize];
		edgeSourceSet = new IntOpenHashSet[initialSize];
		edgeWeights = new ArrayList[initialSize];
		edgeSources = new ArrayList[initialSize];
		this.initialNumEdgesPerNode = initialNumEdgesPerNode;
	}
	
	public int getSize() {
		return nodes.cardinality();
	}
	
	private void ensureCapacity(int minSize) {
		if (minSize > size) {
			int newSize = Math.max(minSize, size * 2);
			nodeClasses = Arrays.copyOf(nodeClasses, newSize);
			edgeSourceSet = Arrays.copyOf(edgeSourceSet, newSize);
			edgeSources = Arrays.copyOf(edgeSources, newSize);
			edgeWeights = Arrays.copyOf(edgeWeights, newSize);
			size = newSize;
		}
	}
	
	public void addNode(Integer node, ArrayList<Integer> sources, ArrayList<Float> weights) {
		ensureCapacity(node + 1);
		nodes.set(node);
		for (int source : sources) {
			addNode(source);
		}
		edgeSourceSet[node] = new IntOpenHashSet(sources.size());
		edgeSourceSet[node].addAll(sources);
		edgeSources[node] = sources;
		edgeWeights[node] = weights;
	}

	public void addNode(Integer node) {
		ensureCapacity(node + 1);
		if (!nodes.get(node)) {
			nodes.set(node);
			edgeSourceSet[node] = new IntOpenHashSet(initialNumEdgesPerNode);
			edgeSources[node] = new ArrayList<Integer>(initialNumEdgesPerNode);
			edgeWeights[node] = new ArrayList<Float>(initialNumEdgesPerNode);
		}
	}
	
	public boolean hasNode(Integer node) {
		return nodes.get(node);
	}
	
	public List<Integer> getNodes() {
		ArrayList<Integer> allNodes = new ArrayList<Integer>(nodes.cardinality());
		for (int node = 0; node < size; node++) {
			if (nodes.get(node))
				allNodes.add(node);
		}
		return allNodes;
	}
	
	public void addEdge(Integer from, Integer to, Float weight) {
		if (from != to) {
			addNode(from);
			addNode(to);
			if (edgeSourceSet[from].add(to)) {
				edgeSources[from].add(to);
				edgeWeights[from].add(weight);
			}
			if (edgeSourceSet[to].add(from)) {
				edgeSources[to].add(from);
				edgeWeights[to].add(weight);
			}
		}
	}
	
	public float getEdgeWeight(Integer from, Integer to) {
		ArrayList<Integer> edges = edgeSources[from];
		ArrayList<Float> weights = edgeWeights[from];
		for (int i = 0; i < edges.size(); i++) {
			if (edges.get(i).equals(to)) {
				return weights.get(i);
			}
		}
		return 0;
	}
	
	public List<Integer> getNeighbors(Integer node) {
		if (edgeSources[node] == null) {
			return Collections.<Integer>emptyList();
		} else {
			return edgeSources[node];
		}
	}
	
	public List<Edge<Integer, Float>> getEdges(Integer node) {
		if (edgeSources[node] == null) {
			return Collections.<Edge<Integer, Float>>emptyList();
		} else {
			List<Integer> sources = edgeSources[node];
			List<Float> weights = edgeWeights[node];
			ArrayList<Edge<Integer, Float>> edges = new ArrayList<Edge<Integer, Float>>(sources.size());
			for (int i = 0; i < sources.size(); i++) {
				edges.add(new Edge<Integer, Float>(sources.get(i), weights.get(i)));
			}
			return edges;
		}
	}
	
	public void sortEdges() {
		for (Integer node = 0; node < edgeSources.length; node++) {
			if (edgeSources[node] == null) continue;
			
			ArrayList<Integer> sources = edgeSources[node];
			ArrayList<Float> weights = edgeWeights[node];
			Integer[] indexes = new Integer[edgeSources[node].size()];
			for (int i = 0; i < indexes.length; i++)
				indexes[i] = i;
			Arrays.sort(indexes, new Comparator<Integer>() {
				@Override
				public int compare(final Integer i1, final Integer i2) {
					return Float.compare(weights.get(i1), weights.get(i2));
				}
			});
			
			ArrayList<Integer> sourcesSorted = new ArrayList<Integer>(sources.size());
			ArrayList<Float> weightsSorted = new ArrayList<Float>(weights.size());
			for (int i = 0; i < indexes.length; i++) {
				sourcesSorted.add(sources.get(indexes[i]));
				weightsSorted.add(weights.get(indexes[i]));
			}
			edgeSources[node] = sourcesSorted;
			edgeWeights[node] = weightsSorted;
		}
	}
	
	public List<Integer> getClosestNeighbors(Integer node, int topn) {
		if (edgeSources[node] == null) {
			return Collections.<Integer>emptyList();
		}
		
		// get the neighbors (copy) and weights
		ArrayList<Integer> neighbors = edgeSources[node];
		ArrayList<Integer> neighborsCopy = new ArrayList<Integer>(neighbors);
		ArrayList<Float> weights = edgeWeights[node];
		// sort neighbors by weights
		Collections.sort(neighborsCopy, new Comparator<Integer>() {
			@Override
			public int compare(final Integer i1, final Integer i2) {
				return Float.compare(weights.get(neighbors.indexOf(i1)), weights.get(neighbors.indexOf(i2)));
			}
		});
		// return topn results
		return neighborsCopy.subList(0, Math.min(topn, neighborsCopy.size()));
	}
	
	public Integer getNodeClass(Integer node) {
		if (node < nodeClasses.length)
			return nodeClasses[node];
		else
			return null;
	}
	
	public void setNodeClass(Integer node, Integer c) {
		if (node < nodeClasses.length)
			nodeClasses[node] = c;
	}
	
}