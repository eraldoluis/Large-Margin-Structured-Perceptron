package br.pucrio.inf.learn.structlearning.discriminative.application.pq;

public class WeightedGraph {
	private Node[] adjList;

	public WeightedGraph(int numberOfVertices) {
		this.adjList = new Node[numberOfVertices];
		
		for(int i = 0; i < numberOfVertices; ++i) {
			Node node = new Node(i, Double.NaN, null);
			this.adjList[i] = node;
		}
	}
	
	public int getNumberOfVertices() {
		return this.adjList.length;
	}
	
	public double getWeight(int source, int target) {
		int numberOfVertices = this.adjList.length;
		if(source > numberOfVertices - 1 || source < 0 ||
		   target > numberOfVertices - 1 || target < 0)
			return Double.NaN;
		
		Node node = this.adjList[source].getNext();
		while(node != null) {
			if(node.getVertex() == target)
				return node.getWeight();
			
			node = node.getNext();
		}
		
		return Double.NaN;
	}
	
	public boolean addEdge(int source, int target, double weight) {
		int numberOfVertices = this.adjList.length;
		if(source > numberOfVertices - 1 || source < 0 ||
		   target > numberOfVertices - 1 || target < 0)
			return false;
		
		Node previousNode = this.adjList[source];
		Node currentNode  = this.adjList[source].getNext();
		while(currentNode != null) {
			if(currentNode.getVertex() == target)
				return false;
			
			previousNode = currentNode;
			currentNode  = currentNode.getNext();
		}
		
		Node newNode = new Node(target, weight, null);
		previousNode.setNext(newNode);
		
		return true;
	}
	
	public boolean removeEdge(int source, int target) {
		int numberOfVertices = this.adjList.length;
		if(source > numberOfVertices - 1 || source < 0 ||
		   target > numberOfVertices - 1 || target < 0)
			return false;
		
		Node previousNode = this.adjList[source];
		Node currentNode  = this.adjList[source].getNext();
		while(currentNode != null) {
			if(currentNode.getVertex() == target) {
				previousNode.setNext(currentNode.getNext());
				return true;
			}
			
			previousNode = currentNode;
			currentNode  = currentNode.getNext();
		}
		
		return false;
	}
	
	public int[] getNeighbors(int vertex) {
		int numberOfNeighbors = 0;
		Node node = this.adjList[vertex].getNext();
		while(node != null) {
			++numberOfNeighbors;
			node = node.getNext();
		}
		
		int[] neighborArray = new int[numberOfNeighbors];
		node = this.adjList[vertex].getNext();
		for(int i = 0; i < numberOfNeighbors; ++i) {
			neighborArray[i] = node.getVertex();
			node = node.getNext();
		}
		
		return neighborArray;
	}
	
	public void printGraph() {
		for(int i = 0; i < this.adjList.length; ++i) {
			System.out.print(i + ": ");
			
			Node node = this.adjList[i].getNext();
			while(node != null) {
				System.out.print(node.getVertex() + "(" + node.getWeight() + ") ");
				node = node.getNext();
			}
			
			System.out.println();
		}
	}
}
