package br.pucrio.inf.learn.structlearning.discriminative.application.pq;

import java.util.ArrayList;

public class Dijkstra {

	public static int[] dijkstra(WeightedGraph graph, int source) {
		/*
		 * Find the shortest path in a graph from source to every node.
		 * 
		 * @param graph: a weighted graph represented by the class
		 * WeightedGraph.
		 * 
		 * @param source: index of the source vertex.
		 * 
		 * @return: an array[graph.getNumberOfVertices()] of predecessors. To
		 * find the shortest path from node i to source, we have to get the
		 * predecessors of i iteratively, until reach node 0. Below we put an
		 * example.
		 * 
		 * int pred[] = Dijkstra.dijkstra(graph, 0); int node = 3;
		 * 
		 * System.out.println(node); while(pred[node] != 0) { node = pred[node];
		 * System.out.println(node); } System.out.println(0);
		 */
		double[] dist = new double[graph.getNumberOfVertices()];
		int[] pred = new int[graph.getNumberOfVertices()];
		boolean[] visited = new boolean[graph.getNumberOfVertices()];

		for (int i = 0; i < dist.length; ++i) {
			dist[i] = Double.MAX_VALUE;
			visited[i] = false;
		}
		dist[source] = 0d;

		for (int i = 0; i < dist.length; ++i) {
			int next = minVertex(dist, visited);
			visited[next] = true;

			int[] neighbors = graph.getNeighbors(next);
			for (int j = 0; j < neighbors.length; ++j) {
				int currrentNeighbor = neighbors[j];
				double d = dist[next] + graph.getWeight(next, currrentNeighbor);

				if (dist[currrentNeighbor] > d) {
					dist[currrentNeighbor] = d;
					pred[currrentNeighbor] = next;
				}
			}
		}

		return pred;
	}

	public static int minVertex(double[] dist, boolean[] visited) {
		double x = Double.MAX_VALUE;
		int y = -1;

		for (int i = 0; i < dist.length; ++i) {
			if (!visited[i] && dist[i] < x) {
				y = i;
				x = dist[i];
			}
		}

		return y;
	}

	public static void printPath(int[] pred, int source, int target) {
		ArrayList<Integer> path = new ArrayList<Integer>();

		int currentVertex = target;
		while (currentVertex != source) {
			path.add(0, currentVertex);
			currentVertex = pred[currentVertex];
		}
		path.add(0, source);

		System.out.println(path);
	}
}
