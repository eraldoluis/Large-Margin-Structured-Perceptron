package br.pucrio.inf.learn.structlearning.discriminative.application.pq;

public class Node {
	private int vertex;
	private double weight;
	private Node next;
	
	public Node(int vertex, double weight, Node next) {
		this.vertex = vertex;
		this.weight = weight;
		this.next = next;
	}
	
	public Node() {
		this(-1, Double.NaN, null);
	}

	public int getVertex() {
		return vertex;
	}

	public void setVertex(int vertex) {
		this.vertex = vertex;
	}

	public double getWeight() {
		return weight;
	}

	public void setWeight(double weight) {
		this.weight = weight;
	}

	public Node getNext() {
		return next;
	}

	public void setNext(Node next) {
		this.next = next;
	}
}
