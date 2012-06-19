package br.pucrio.inf.learn.util.maxbranching;

import java.util.Arrays;
import java.util.LinkedList;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;

/**
 * Chu-Liu-Edmonds' algorithm for finding a maximum branching in a directed
 * graph. This implementation is optimized for dense graphs, i.e., number of
 * edges is Big Omega of n*n, where n is the number of nodes.
 * 
 * This implementation is based on Tarjan's 'Finding Optimum Branchings' paper.
 * It also uses the improved priority queue representation for dense graphs
 * suggested in that paper.
 * 
 * This implementation is optimized to solve a long sequence of instances with a
 * bounded dimension in order to avoid reallocation of memory. So, the user must
 * give the size (number of nodes) of the biggest instance when allocating an
 * object of this class.
 * 
 * @author eraldo
 * 
 */
public class MaximumBranchingAlgorithm {
	/**
	 * Logging object.
	 */
	private final static Log LOG = LogFactory
			.getLog(MaximumBranchingAlgorithm.class);

	/**
	 * Union-find data structure to store the partition of the strongly
	 * connected components (SCCs).
	 */
	private DisjointSets sPartition;

	/**
	 * Union-find data structure to store the partition of the weakly connected
	 * components (WCCs).
	 */
	private DisjointSets wPartition;

	/**
	 * Priority queue of incoming edges for each SCC. This implementation is
	 * adequate for dense graphs.
	 * 
	 * The array <code>incomingEdges[scc]</code> stores the incoming edges for
	 * SCC <code>scc</code> that were not consired yet. The value
	 * <code>incomingEdges[scc][from]</code> stores the end node (which lies
	 * within the SCC <code>scc</code>) of the edge whose start point is
	 * <code>from</code>. If no edge exists from the start node
	 * <code>from</code>, this value will be -1.
	 * 
	 * Since, for each SCC, there always will be at most one incoming edge from
	 * the same start node, this data structure is more efficient for dense
	 * graphs than keeping a generic priority queue.
	 */
	private int[][] incomingEdges;

	/**
	 * Origin node of the unique incoming edge for each SCC.
	 */
	private int[] enterFromNode;

	/**
	 * Destin node of the unique incoming edge for each SCC.
	 */
	private int[] enterToNode;

	/**
	 * For each SCC, store the destin node whose incoming edge has the minimum
	 * weight within the current branching. If such edge exists (it exists only
	 * for cycles within the current branching), it is the one removed when a
	 * outside incoming edge is included in the branching.
	 */
	private int[] min;

	/**
	 * Edges in the maximum branching. It may include additional edges that form
	 * cycles but those are directly removed when one walks through the
	 * branching by always starting from <code>min</code> nodes.
	 */
	private boolean[][] maxBranching;

	/**
	 * Auxiliar array to graph search.
	 */
	private boolean[] visited;

	/**
	 * If <code>true</code>, then emit a warn whenever the predicted branching
	 * comprises more than one root node.
	 */
	private boolean checkUniqueRoot;

	/**
	 * Specify whether negative edges are included when no other option exists.
	 */
	private boolean onlyPositiveEdges;

	/**
	 * Allocate data structures to deal with the given maximum number of nodes.
	 * 
	 * @param maxNumberOfNodes
	 */
	public MaximumBranchingAlgorithm(int maxNumberOfNodes) {
		realloc(maxNumberOfNodes);
		checkUniqueRoot = true;
	}

	/**
	 * Realloc internal data structures to fit the given maximum size needed.
	 * 
	 * @param maxNumberOfNodes
	 */
	public void realloc(int maxNumberOfNodes) {
		sPartition = new DisjointSets(maxNumberOfNodes);
		wPartition = new DisjointSets(maxNumberOfNodes);
		incomingEdges = new int[maxNumberOfNodes][maxNumberOfNodes];
		enterFromNode = new int[maxNumberOfNodes];
		enterToNode = new int[maxNumberOfNodes];
		min = new int[maxNumberOfNodes];
		maxBranching = new boolean[maxNumberOfNodes][maxNumberOfNodes];
		visited = new boolean[maxNumberOfNodes];
	}

	/**
	 * Set whether the prediction algorithm checks if the predicted branching
	 * comprises exactly one root node. If this flag is <code>true</code>, then
	 * emit a warn whenever the predicted branching comprises more than one root
	 * node.
	 * 
	 * @param check
	 */
	public void setCheckUniqueRoot(boolean check) {
		checkUniqueRoot = check;
	}

	/**
	 * Fill <code>maxBranching</code> with a maximum branching of the given
	 * graph <code>graph</code> and rooted in the given node.
	 * 
	 * @param numberOfNodes
	 * @param graph
	 * @param invertedMaxBranching
	 * @return the weight of the built solution.
	 */
	public double findMaxBranching(int numberOfNodes, double[][] graph,
			int[] invertedMaxBranching) {
		// Maximum branching is initially empth.
		for (int from = 0; from < numberOfNodes; ++from)
			Arrays.fill(maxBranching[from], 0, numberOfNodes, false);

		// Partitions initially comprise isolated nodes.
		sPartition.clear(numberOfNodes);
		wPartition.clear(numberOfNodes);

		/*
		 * List of root components, i.e., SCCs that have no incoming edges
		 * (enter[scc] == null). In the beginning, every SCC is a root
		 * component.
		 */
		LinkedList<Integer> rootComponents = new LinkedList<Integer>();

		// Initialize the remaining data structures.
		for (int scc = 0; scc < numberOfNodes; ++scc) {

			// Initially, every SCC (node) is a root component.
			rootComponents.add(scc);

			// The head of its root component is its only node.
			min[scc] = scc;

			// No edge entering any SCC.
			enterFromNode[scc] = -1;
			enterToNode[scc] = -1;

			/*
			 * Add all incoming edges of <code>scc</code> to its SCC priority
			 * queue.
			 */
			for (int from = 0; from < numberOfNodes; ++from) {
				if (Double.isNaN(graph[from][scc]))
					// Edge does not exist.
					incomingEdges[scc][from] = -1;
				else
					incomingEdges[scc][from] = scc;
			}

			// Remove autocycle edges.
			incomingEdges[scc][scc] = -1;
		}

		// Root components with no available incoming edges.
		LinkedList<Integer> doneRootComponents = new LinkedList<Integer>();

		while (!rootComponents.isEmpty()) {
			// Get some arbitrary root component.
			int sccTo = rootComponents.pop();

			// Find the maximum edge entering the component 'sccTo'.
			int maxInEdgeFromNode = -1;
			double maxInEdgeWeight = Double.NEGATIVE_INFINITY;
			for (int from = 0; from < numberOfNodes; ++from) {
				int inEdgeToNode = incomingEdges[sccTo][from];
				if (inEdgeToNode == -1)
					continue;
				double w = graph[from][inEdgeToNode];
				if (Double.isNaN(w))
					LOG.warn("Using NaN edge: " + from + ">" + inEdgeToNode);

				if (w > maxInEdgeWeight && (!onlyPositiveEdges || w >= 0d)) {
					maxInEdgeFromNode = from;
					maxInEdgeWeight = w;
				}
			}

			if (maxInEdgeFromNode == -1) {
				// No edge left to consider in this component. So, it is done.
				doneRootComponents.add(sccTo);
				continue;
			}

			/*
			 * Get the end node of the selected edge and remove the edge from
			 * the SCC priority queue.
			 */
			int maxInEdgeToNode = incomingEdges[sccTo][maxInEdgeFromNode];
			incomingEdges[sccTo][maxInEdgeFromNode] = -1;

			// SCC component of the start node of the maximum edge.
			int sccFrom = sPartition.find(maxInEdgeFromNode);

			if (sccFrom == sccTo) {
				/*
				 * Intern edge. Disconsider this edge but add its component to
				 * be considered again later.
				 */
				rootComponents.add(sccTo);
				continue;
			}

			// Include the selected edge in the current branching.
			maxBranching[maxInEdgeFromNode][maxInEdgeToNode] = true;

			// WCC component of the start node.
			int wssFrom = wPartition.find(maxInEdgeFromNode);
			// WCC component of the end node.
			int wssTo = wPartition.find(maxInEdgeToNode);

			// Edge connects two different WCCs.
			if (wssFrom != wssTo) {
				// Unite the two WCCs.
				wPartition.union(wssFrom, wssTo);
				// Store the current entering edge of the selected SCC.
				enterFromNode[sccTo] = maxInEdgeFromNode;
				enterToNode[sccTo] = maxInEdgeToNode;
				continue;
			}

			/*
			 * Edge is within the same WCC, thus its inclusion will create a new
			 * SCC by uniting some old SCCs (the ones on the path from
			 * maxInEdgeToNode to maxInEdgeFromNode).
			 * 
			 * First, find the minimum edge to be removed among all SCCs that
			 * will be united.
			 */
			double minEdgeWeight = Double.POSITIVE_INFINITY;
			int minScc = -1;
			int tmpEdgeFromNode = maxInEdgeFromNode;
			int tmpEdgeToNode = maxInEdgeToNode;
			while (tmpEdgeFromNode != -1) {
				double tmpEdgeWeight = graph[tmpEdgeFromNode][tmpEdgeToNode];
				if (tmpEdgeWeight < minEdgeWeight) {
					minEdgeWeight = tmpEdgeWeight;
					minScc = sPartition.find(tmpEdgeToNode);
				}

				// Next edge.
				int tmpScc = sPartition.find(tmpEdgeFromNode);
				tmpEdgeFromNode = enterFromNode[tmpScc];
				tmpEdgeToNode = enterToNode[tmpScc];
			}

			// Set the head of the current SCC.
			min[sccTo] = min[minScc];

			// Increment incoming edges weights.
			double inc = minEdgeWeight - maxInEdgeWeight;
			for (int from = 0; from < numberOfNodes; ++from) {
				int to = incomingEdges[sccTo][from];
				if (to == -1)
					continue;
				graph[from][to] += inc;
			}

			// Include all used SCCs in the current SCC 'sccTo'.
			tmpEdgeFromNode = enterFromNode[sccFrom];
			tmpEdgeToNode = enterToNode[sccFrom];
			while (tmpEdgeFromNode != -1) {
				/*
				 * Increment incoming edges weight and include them in the
				 * current SCC priority queue.
				 */
				int tmpSccTo = sPartition.find(tmpEdgeToNode);
				double tmpEdgeWeight = graph[tmpEdgeFromNode][tmpEdgeToNode];
				inc = minEdgeWeight - tmpEdgeWeight;
				for (int from = 0; from < numberOfNodes; ++from) {
					int tmpTo = incomingEdges[tmpSccTo][from];
					if (tmpTo == -1)
						continue;
					graph[from][tmpTo] += inc;
					int to = incomingEdges[sccTo][from];
					if (to == -1 || graph[from][tmpTo] > graph[from][to])
						incomingEdges[sccTo][from] = tmpTo;
				}

				// Unite the two SCCs.
				sPartition.union(sccTo, tmpSccTo);

				// Next edge.
				int tmpScc = sPartition.find(tmpEdgeFromNode);
				tmpEdgeFromNode = enterFromNode[tmpScc];
				tmpEdgeToNode = enterToNode[tmpScc];
			}

			// Include the new SCC to be considered in the future.
			rootComponents.add(sccTo);
		}

		if (checkUniqueRoot && doneRootComponents.size() > 1)
			LOG.warn("Final root components list contains more than one element");

		// Invert the maximum branching and compute its weight.
		double weight = 0d;
		Arrays.fill(visited, 0, numberOfNodes, false);
		Arrays.fill(invertedMaxBranching, 0, numberOfNodes, -1);
		for (int scc : doneRootComponents)
			weight += invertBranching(numberOfNodes, graph, min[scc],
					maxBranching, visited, invertedMaxBranching);

		// Return the weight of the built solution.
		return weight;
	}

	// /**
	// * Print graph weights to the error output stream.
	// *
	// * @param numberOfNodes
	// * @param graph
	// * @param root
	// */
	// private void printGraph(int numberOfNodes, double[][] graph, int root) {
	// System.err.println();
	// System.err.println("Graph weights");
	// for (int from = 0; from < numberOfNodes; ++from) {
	// System.err.print(from);
	// for (int to = 0; to < numberOfNodes; ++to) {
	// System.err.print(String.format("\t%3d:%10f", to,
	// graph[from][to]));
	// }
	// System.err.println();
	// }
	// System.err.println();
	// }

	/**
	 * Walk through the given branching from <code>node</code> and store the
	 * inverted branching in <code>invertedMaxBranching</code>.
	 * 
	 * In fact, the given branching can include cycles. Thus, we use the array
	 * <code>visited</code> and disconsider the last edges of each cycle.
	 * 
	 * @param numberOfNodes
	 * @param from
	 * @param branching
	 * @param visited
	 * @param invertedBranching
	 * @return the weight of the subtree under <code>node</code>.
	 */
	private double invertBranching(final int numberOfNodes,
			final double[][] graph, int from, boolean[][] branching,
			boolean[] visited, int[] invertedBranching) {
		double weight = 0d;
		visited[from] = true;
		for (int to = 0; to < numberOfNodes; ++to) {
			if (!branching[from][to] || visited[to])
				continue;
			invertedBranching[to] = from;
			weight += graph[from][to];
			weight += invertBranching(numberOfNodes, graph, to, branching,
					visited, invertedBranching);
		}

		return weight;
	}

	/**
	 * Set whether the algorithm is allowed to include edges with negative
	 * weight when no nonnegative edge exists.
	 * 
	 * @param val
	 */
	public void setOnlyPositiveEdges(boolean val) {
		this.onlyPositiveEdges = val;
	}

	/**
	 * Test program for the maximum branching algorithm.
	 * 
	 * @param args
	 */
	public static void main(String[] args) {

		double ifn = Double.NEGATIVE_INFINITY;

		double[][] weights = { { ifn, 100, 400, 100 }, { ifn, ifn, 100, ifn },
				{ ifn, 25, ifn, 75 }, { ifn, ifn, 300, ifn }, };

		MaximumBranchingAlgorithm eds = new MaximumBranchingAlgorithm(4);
		int[] maxBranching = new int[weights.length];
		eds.findMaxBranching(4, weights, maxBranching);

		// Print maximum branching per node.
		System.out.println("Maximum branching:");
		for (int to = 1; to < maxBranching.length; ++to)
			System.out.println(to + " <- " + maxBranching[to]);
	}
}
