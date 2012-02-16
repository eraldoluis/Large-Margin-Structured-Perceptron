package br.pucrio.inf.learn.structlearning.discriminative.application.pq;

import java.util.ArrayList;

public class Dijkstra {

	public static int[] dijkstra(WeightedGraph graph, int source) {
		double[] dist     = new double[graph.getNumberOfVertices()];
		int[] pred        = new int[graph.getNumberOfVertices()];
		boolean[] visited = new boolean[graph.getNumberOfVertices()];
		
		for(int i = 0; i < dist.length; ++i) {
			dist[i]    = Double.MAX_VALUE;
			visited[i] = false;
		}
		dist[source] = 0d;
		
		for(int i = 0; i < dist.length; ++i) {
			int next = minVertex(dist, visited);
			visited[next] = true;
			
			int[] neighbors = graph.getNeighbors(next);
			for(int j = 0; j < neighbors.length; ++j) {
				int currrentNeighbor = neighbors[j];
				double d = dist[next] + graph.getWeight(next, currrentNeighbor);
				
				if(dist[currrentNeighbor] > d) {
					dist[currrentNeighbor] = d;
					pred[currrentNeighbor] = next;
				}
			}
		}
		
		return pred;
	}
	
	public static int minVertex(double[] dist, boolean[] visited) {
		double x = Double.MAX_VALUE;
		int y    = -1;
		
		for(int i = 0; i < dist.length; ++i) {
			if(!visited[i] && dist[i] < x) {
				y = i;
				x = dist[i];
			}
		}
		
		return y;
	}
	
	public static void printPath(int[] pred, int source, int target) {
		ArrayList path = new ArrayList();
		
		int currentVertex = target;
		while(currentVertex != source) {
			path.add(0, currentVertex);
			currentVertex = pred[currentVertex];
		}
		path.add(0, source);
		
		System.out.println(path);
	}
}
