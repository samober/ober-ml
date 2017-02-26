package com.ober.ml.cw;

public class ClusterMember {
	
	private Integer node;
	private Float weight;
	
	public ClusterMember() {
		this.node = 0;
		this.weight = 0f;
	}
	
	public ClusterMember(Integer node, Float weight) {
		this.node = node;
		this.weight = weight;
	}
	
	public Integer getNode() {
		return node;
	}
	
	public void setNode(Integer node) {
		this.node = node;
	}
	
	public Float getWeight() {
		return weight;
	}
	
	public void setWeight(Float weight) {
		this.weight = weight;
	}
	
	@Override
	public boolean equals(Object other) {
		if (other == null) return false;
		if (other == this) return true;
		if (!(other instanceof ClusterMember))
			return false;
			
		ClusterMember member = (ClusterMember) other;
		return member.node == node;
	}
	
	@Override
	public int hashCode() {
		final int prime = 31;
		int result = 1;
		result = prime * result + node;
		return result;
	}
	
}
