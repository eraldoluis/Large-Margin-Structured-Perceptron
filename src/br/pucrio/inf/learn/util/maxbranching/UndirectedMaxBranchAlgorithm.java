package br.pucrio.inf.learn.util.maxbranching;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;

/**
 * Implement a maximum branching algorithm for undirected graphs. A directed
 * graph is given as input and the algorithm selects only one directed edge to
 * represent the undirected edge. For two nodes i and j such that i < j, it
 * tries first the edge (i,j). If this edge doest not exist, it uses the (j,i)
 * edge
 * 
 * @author eraldo
 * 
 */
public class UndirectedMaxBranchAlgorithm implements MaximumBranchingAlgorithm {

	/**
	 * List of all edges.
	 */
	private ArrayList<SimpleWeightedEdge> edges;

	/**
	 * Used to walk through the spanning tree.
	 */
	private boolean[] visited;

	/**
	 * Incidence matrix that represents the maximum spanning tree. Only the
	 * superior diagonal matrix is used.
	 */
	private boolean[][] incidence;

	/**
	 * Connected components.
	 */
	private DisjointSets partition;

	/**
	 * Whether to avoid negative-weight edges.
	 */
	private boolean onlyPositiveEdges;

	/**
	 * Create a undirected maximum branching algorithm (Kruskal) that is able to
	 * handle graphs with up to the given <code>maxNumberOfNodes</code> nodes.
	 * 
	 * @param maxNumberOfNodes
	 */
	public UndirectedMaxBranchAlgorithm(int maxNumberOfNodes) {
		realloc(maxNumberOfNodes);
	}

	@Override
	public double findMaxBranching(int numberOfNodes, double[][] graph,
			int[] invertedMaxBranching) {
		// Clear the list of all edges.
		edges.clear();
		// Add edges with their weights to the list.
		for (int from = 0; from < numberOfNodes; ++from) {
			for (int to = 0; to < numberOfNodes; ++to) {
				if (from == to)
					// Skip auto-cycle arcs.
					continue;
				if (!Double.isNaN(graph[from][to])
						&& (!onlyPositiveEdges || graph[from][to] >= 0d))
					edges.add(new SimpleWeightedEdge(from, to, graph[from][to]));
			}
		}
		// Sort edges by weight.
		Collections.sort(edges);

		// Initialize the disjoint sets (one component for each node).
		partition.clear(numberOfNodes);

		for (int from = 0; from < numberOfNodes; ++from)
			Arrays.fill(incidence[from], from + 1, numberOfNodes, false);

		// Greedily select edges while avoiding cycles.
		double weight = 0d;
		for (SimpleWeightedEdge edge : edges) {
			// Component id of the source node.
			int pFrom = partition.find(edge.from);
			// Component id of the target node.
			int pTo = partition.find(edge.to);
			// Add edge if it does not create a cycle.
			if (pFrom != pTo) {
				// Connect the two components.
				partition.union(pFrom, pTo);
				// Add edge to the tree.
				incidence[edge.from][edge.to] = true;
				// Account for its weight.
				weight += edge.weight;
			}
		}

		// Fill the output array with an empty tree (-1 for all incoming edges).
		Arrays.fill(invertedMaxBranching, 0, numberOfNodes, -1);

		/*
		 * Orient edges to allow its representation in the incoming edges array.
		 * This process may include inexistent edges (as of the used
		 * orientation).
		 */
		Arrays.fill(visited, 0, numberOfNodes, false);
		for (int from = 0; from < numberOfNodes; ++from)
			if (!visited[from])
				orient(from, numberOfNodes, invertedMaxBranching);

		// Return the weight of the created tree.
		return weight;
	}

	/**
	 * Fill the inverted branching array (incoming edges array) by performing an
	 * orientation in the tree given by the <code>incidence</code> matrix.
	 * 
	 * @param node
	 * @param numberOfNodes
	 * @param invertedBranching
	 */
	private void orient(int node, int numberOfNodes, int[] invertedBranching) {
		// Flag this node as visited.
		visited[node] = true;

		// Consider arcs leaving the given node (node).
		for (int to = 0; to < numberOfNodes; ++to) {
			if (node == to)
				// Skip auto-cycle arcs.
				continue;

			if (!visited[to] && incidence[node][to]) {
				invertedBranching[to] = node;
				orient(to, numberOfNodes, invertedBranching);
			}
		}

		// Consider arcs arriving at the given node (node).
		for (int from = 0; from < numberOfNodes; ++from) {
			if (node == from)
				// Skip auto-cycle arcs.
				continue;

			if (!visited[from] && incidence[from][node]) {
				invertedBranching[from] = node;
				orient(from, numberOfNodes, invertedBranching);
			}
		}
	}

	@Override
	public void realloc(int maxNumberOfNodes) {
		if (visited != null && visited.length >= maxNumberOfNodes)
			return;
		partition = new DisjointSets(maxNumberOfNodes);
		edges = new ArrayList<SimpleWeightedEdge>(maxNumberOfNodes
				* (maxNumberOfNodes - 1));
		visited = new boolean[maxNumberOfNodes];
		incidence = new boolean[maxNumberOfNodes][maxNumberOfNodes];
	}

	@Override
	public void setCheckUniqueRoot(boolean check) {
	}

	@Override
	public boolean isCheckUniqueRoot() {
		return false;
	}

	@Override
	public void setOnlyPositiveEdges(boolean val) {
		onlyPositiveEdges = val;
	}

	@Override
	public boolean isOnlyPositiveEdges() {
		return onlyPositiveEdges;
	}

	/**
	 * Test the undirected maximum branching algorithm.
	 * 
	 * @param args
	 */
	public static void main(String[] args) {
		UndirectedMaxBranchAlgorithm alg = new UndirectedMaxBranchAlgorithm(5);
		double[][] graph = {
				//
				{ 0, 10, 0, 0, 10 }, //
				{ 0, 0, 1, 5, 10 },//
				{ 0, 0, 0, 2, 20 }, //
				{ 0, 0, 0, 0, 5 }, //
				{ 0, 0, 0, 0, 0 }, //
		};
		int[] tree = new int[5];

		alg.findMaxBranching(5, graph, tree);

		printBranching(tree, 5);
	}

	/**
	 * Print the given tree.
	 * 
	 * @param branching
	 * @param numNodes
	 */
	private static void printBranching(int[] branching, int numNodes) {
		for (int node = 0; node < numNodes; ++node)
			System.out.print("(" + branching[node] + "," + node + ") ");
		System.out.println();
	}
}
