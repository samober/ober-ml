package com.ober.ml.cw;

import java.util.HashSet;
import java.util.Set;

public class Cluster {
	
	private Integer node;
	private Integer sense;
	private HashSet<ClusterMember> members;
	
	public Cluster() {
		this(0, 0);
	}
	
	public Cluster(Integer node, Integer sense) {
		this.node = node;
		this.sense = sense;
		this.members = new HashSet<ClusterMember>();
	}
	
	public Integer getNode() {
		return node;
	}
	
	public void setNode(Integer node) {
		this.node = node;
	}
	
	public Integer getSense() {
		return sense;
	}
	
	public void setSense(Integer sense) {
		this.sense = sense;
	}
	
	public Set<ClusterMember> getMembers() {
		return members;
	}
	
	public void clearMembers() {
		members.clear();
	}
	
	public void addMember(Integer node, Float weight) {
		members.add(new ClusterMember(node, weight));
	}
	
	public void addMember(ClusterMember member) {
		members.add(member);
	}
	
}