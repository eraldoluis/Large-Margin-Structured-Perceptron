package br.pucrio.inf.learn.structlearning.discriminative.application.pq;

import java.util.ArrayList;

public class PerfectBipartiteMatching {
	
	public static ArrayList<int[]> perfectBipartiteMatching(double[][] hungarianCostMatrix) {
		/*
		 * Find a perfect bipartite matching, given a square non-negative cost matrix.
		 * 
		 * @param hungarianCostMatrix: a square cost matrix with non-negative values.
		 * 
		 * @return: an ArrayList<int[2]> with arcs that belong to perfect bipartite
		 * 			matching.
		 */
		int hcmSize = hungarianCostMatrix.length;
		
		double[] pLine   = new double[hcmSize];
		double[] pColumn = new double[hcmSize];
		
		//Initialize both p arrays
		for(int i = 0; i < hcmSize; ++i) {
			double maxLine = Double.MIN_VALUE;
			for(int j = 0; j < hcmSize; ++j) {
				if(hungarianCostMatrix[i][j] > maxLine)
					maxLine = hungarianCostMatrix[i][j];
			}
			pLine[i]   = maxLine;
			pColumn[i] = 0d;
		}
		
		//Initial perturbation
		for(int i = 0; i < hcmSize; ++i)
			for(int j = 0; j < hcmSize; ++j) {
				hungarianCostMatrix[i][j] += - pLine[i] + pColumn[j];
				hungarianCostMatrix[i][j] *= -1;
			}
		
		//Create a graph with artificial source and target
		int numberOfVertices = hcmSize * 2 + 2;
		WeightedGraph graph = new WeightedGraph(numberOfVertices);
		//Create edges from source to all vertices in the first layer
		for(int i = 1; i < hcmSize + 1; ++i)
			graph.addEdge(0, i, 0d);
		//Create edges from all vertices in the second layer to target
		for(int i = hcmSize + 1; i < numberOfVertices - 1; ++i)
			graph.addEdge(i, numberOfVertices - 1, 0d);
		//Create edges corresponding to hungarian matrix costs
		for(int i = 0; i < hcmSize; ++i)
			for(int j = 0; j < hcmSize; ++j)
				graph.addEdge(i + 1, j + hcmSize + 1, hungarianCostMatrix[i][j]);
		
		//In each iteration, the matching augments by 1. By the end of
		//the iterations, we will have a perfect matching
		ArrayList<int[]> P = new ArrayList<int[]>();
		for(int n = 0; n < hcmSize; ++n) {
			//Run Dijkstra's algorithm
			int pred[] = Dijkstra.dijkstra(graph, 0);
			
			//Build array 'd' with costs from source to each node
			int currentVertex;
			int previousVertex;
			double[] d = new double[numberOfVertices];
			for(int i = 0; i < numberOfVertices; ++i) {
				currentVertex = i;
				previousVertex = pred[i];
				
				double cost = 0d;
				while(previousVertex != 0) {
					cost += graph.getWeight(previousVertex, currentVertex);
					currentVertex = previousVertex;
					previousVertex = pred[previousVertex];
				}
				
				if(currentVertex != 0)
					cost += graph.getWeight(previousVertex, currentVertex);
				
				d[i] = cost;
			}
			
			//Put arcs of P (that are inverted) in the correct order and
			//create arcs that link the involved nodes to the artificial
			//source and target
			int[] edge;
			int pSize = P.size();
			for(int i = 0; i < pSize; ++i) {
				edge = P.get(i);
				double weight = graph.getWeight(edge[1], edge[0]);
				
				//Invert edge
				graph.removeEdge(edge[1], edge[0]);
				graph.addEdge(edge[0], edge[1], weight);
				//Connect vertices from source and target
				graph.addEdge(0, edge[0], 0d);
				graph.addEdge(edge[1], numberOfVertices - 1, 0d);
			}
			
			//Build the alternating path A
			ArrayList<int[]> A = new ArrayList<int[]>();
			currentVertex  = pred[numberOfVertices - 1];
			previousVertex = pred[currentVertex];
			while(previousVertex != 0) {
				//Create an edge and add it in the alternating path
				edge    = new int[2];
				edge[0] = previousVertex;
				edge[1] = currentVertex;
				A.add(0, edge);
				
				currentVertex = previousVertex;
				previousVertex = pred[previousVertex];
			}
			
			//Generate a new P, P <- (P-A) U (A-P)
			ArrayList<int[]> newP = new ArrayList<int[]>();
			//(P-A)
			for(int i = 0; i < pSize; ++i) {
				edge = P.get(i);
				if(!A.contains(edge))
					newP.add(edge);
			}
			//(A-P)
			int aSize = A.size();
			for(int i = 0; i < aSize; ++i) {
				edge = A.get(i);
				if(!P.contains(edge))
					newP.add(edge);
			}
			//Update P with newP
			P = new ArrayList<int[]>();
			int newPSize = newP.size(); 
			for(int i = 0; i < newPSize; ++i) {
				edge = newP.get(i);
				int[] newEdge = new int[2];
				newEdge[0] = edge[0];
				newEdge[1] = edge[1];
				P.add(newEdge);
			}
			
			//Invert edges that belong to P and disconnect the vertices from source and target
			pSize = P.size();
			for(int i = 0; i < pSize; ++i) {
				edge = P.get(i);
				double weight = graph.getWeight(edge[0], edge[1]);
				
				//Invert edge
				graph.removeEdge(edge[0], edge[1]);
				graph.addEdge(edge[1], edge[0], weight);
				//Disconnect vertices from source and target
				graph.removeEdge(0, edge[0]);
				graph.removeEdge(edge[1], numberOfVertices - 1);
			}
			
			//Update graph weights - only updates forward arcs
			for(int i = 0; i < hcmSize; ++i)
				for(int j = 0; j < hcmSize; ++j) {
					double currentWeight = graph.getWeight(i + 1, j + 1 + hcmSize);
					if(currentWeight != Double.NaN)
						graph.setWeight(i + 1, j + 1 + hcmSize, currentWeight + d[i + 1] - d[j + 1 + hcmSize]);
				}
		}
		
		//Convert arcs to hungarian matrix representation
		int[] edge;
		int pSize = P.size();
		for(int i = 0; i < pSize; ++i) {
			edge = P.get(i);
			edge[0] = edge[0] - 1;
			edge[1] = edge[1] - 1 - hcmSize;
			P.set(i, edge);
		}
		
		return P;
	}
}
