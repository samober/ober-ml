package com.ober.ml.cw.graph;

public class Edge<N, E> {
	
	private N node;
	private E weight;
	
	public Edge() {
		node = null;
		weight = null;
	}
	
	public Edge(N node, E weight) {
		this.node = node;
		this.weight = weight;
	}
	
	public N getNode() {
		return node;
	}
	
	public E getWeight() {
		return weight;
	}
	
	@Override
	public String toString() {
		return "Edge(node=" + node.toString() + ",weight=" + weight.toString() + ")";
	}
	
	@Override
	public boolean equals(Object other) {
		if (this == other) return true;
		if (other == null || getClass() != other.getClass()) return false;
		Edge edge = (Edge) other;
		return node.equals(edge.node) && weight.equals(edge.weight);
	}
	
	@Override
	public int hashCode() {
		final int prime = 31;
		int result = 1;
		result = prime * result + ((node == null) ? 0 : node.hashCode());
		result = prime * result + ((weight == null) ? 0 : weight.hashCode());
		return result;
	}
	
}