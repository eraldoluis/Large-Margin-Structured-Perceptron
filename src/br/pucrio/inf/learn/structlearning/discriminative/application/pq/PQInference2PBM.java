package br.pucrio.inf.learn.structlearning.discriminative.application.pq;

import br.pucrio.inf.learn.structlearning.discriminative.application.pq.data.Quotation;
import br.pucrio.inf.learn.structlearning.discriminative.application.pq.data.PQInput2;
import br.pucrio.inf.learn.structlearning.discriminative.application.pq.data.PQOutput2;
import br.pucrio.inf.learn.structlearning.discriminative.application.pq.PQModel2;
import br.pucrio.inf.learn.structlearning.discriminative.data.ExampleInput;
import br.pucrio.inf.learn.structlearning.discriminative.data.ExampleOutput;
import br.pucrio.inf.learn.structlearning.discriminative.task.Inference;
import br.pucrio.inf.learn.structlearning.discriminative.task.Model;

import java.util.ArrayList;
import java.util.LinkedList;
import java.util.Collections;
import java.util.Comparator;
import java.util.Iterator;

// Perfect Bipartite Matching
public class PQInference2PBM implements Inference {

	@Override
	public void inference(Model model, ExampleInput input, ExampleOutput output) {
		// TODO Auto-generated method stub
		inference((PQModel2) model, (PQInput2) input, (PQOutput2) output);
	}

	@Override
	public void lossAugmentedInference(Model model, ExampleInput input,
			ExampleOutput referenceOutput, ExampleOutput predictedOutput,
			double lossWeight) {
		lossAugmentedInference((PQModel2) model, (PQInput2) input, (PQOutput2) referenceOutput,
							   (PQOutput2) predictedOutput, lossWeight);
	}
	
	//Perfect Bipartite Matching
	public void inference(PQModel2 model, PQInput2 input, PQOutput2 output) {
		// TODO
	}
	
	//Perfect Bipartite Matching
	public void lossAugmentedInference(PQModel2 model, PQInput2 input,
			PQOutput2 referenceOutput, PQOutput2 predictedOutput,
			double lossWeight) {
		double[][] costMatrix = generateCostMatrix(model, input, referenceOutput, lossWeight);
		double[][] hungarianCostMatrix = preProcessCostMatrix(costMatrix);
		
		int hcmSize = hungarianCostMatrix.length;
		
		double[] pLine   = new double[hcmSize];
		double[] pColumn = new double[hcmSize];
		
		double[][] modifiedHungarianCostMatrix = new double[hcmSize][hcmSize]; 
		
		//Initialize both p arrays
		for(int i = 0; i < hcmSize; ++i) {
			double minLine = Double.MAX_VALUE;
			for(int j = 0; j < hcmSize; ++j) {
				modifiedHungarianCostMatrix[i][j] = hungarianCostMatrix[i][j];
				if(hungarianCostMatrix[i][j] < minLine)
					minLine = hungarianCostMatrix[i][j];
			}
			pLine[i]   = 0d;
			pColumn[i] = minLine;
		}
		
		ArrayList<int[]> M = new ArrayList<int[]>();
		//In each iteration, the matching augments of 1. By the end of
		//the iterations, we will have a perfect matching
		for(int n = 0; n < hcmSize; ++n) {
			//Update hungarian cost matrix
			for(int i = 0; i < hcmSize; ++i)
				for(int j = 0; j < hcmSize; ++j)
					modifiedHungarianCostMatrix[i][j] = hungarianCostMatrix[i][j] + pLine[i] - pColumn[j];
			
			//Create a graph with artificial source and target
			int numberOfVertices = hcmSize * 2 + 2;
			WeightedGraph graph = new WeightedGraph(numberOfVertices);
			//Create edges from source to all vertices in the first layer
			for(int i = 1; i < hcmSize + 1; ++i)
				graph.addEdge(0, i, 0d);
			//Create edges from all vertices in the second layer to target
			for(int i = hcmSize + 1; i < numberOfVertices - 1; ++i)
				graph.addEdge(i, numberOfVertices - 1, 0d);
			//Create edges corresponding to the hungarian matrix costs
			for(int i = 0; i < hcmSize; ++i)
				for(int j = 0; j < hcmSize; ++j)
					graph.addEdge(i + 1, j + hcmSize + 1, modifiedHungarianCostMatrix[i][j]);
			
			//Invert edges that belong to M and disconnect the vertices from source and target
			int mSize = M.size();
			for(int i = 0; i < mSize; ++i) {
				int[] edge = M.get(i);
				double weight = graph.getWeight(edge[0], edge[1]);
				
				//Invert edge
				graph.removeEdge(edge[0], edge[1]);
				graph.addEdge(edge[1], edge[0], weight);
				//Disconnect vertices from source and target
				graph.removeEdge(0, edge[0]);
				graph.removeEdge(edge[1], numberOfVertices - 1);
			}
			
			//Run Dijkstra's algorithm
			int pred[] = Dijkstra.dijkstra(graph, 0);
			
			//Build the alternating path A
			ArrayList<int[]> A = new ArrayList<int[]>();
			int currentVertex  = numberOfVertices - 1;
			int previousVertex = pred[currentVertex];
			int[] edge;
			
			while(previousVertex != 0) {
				//Create an edge and add it in the alternating path
				edge    = new int[2];
				edge[0] = previousVertex;
				edge[1] = currentVertex;
				A.add(0, edge);
				
				currentVertex = previousVertex;
				previousVertex = pred[previousVertex];
			}
			
			//Create the first edge and add it in the alternating path
			edge    = new int[2];
			edge[0] = previousVertex;
			edge[1] = currentVertex;
			A.add(0, edge);
			
			//Generate a new M, M <- (M-A) U (A-M)
			ArrayList<int[]> newM = new ArrayList<int[]>();
			//(M-A)
			for(int i = 0; i < mSize; ++i) {
				edge = M.get(i);
				if(!A.contains(edge))
					newM.add(edge);
			}
			//(A-M)
			int aSize = A.size();
			for(int i = 0; i < aSize; ++i) {
				edge = A.get(i);
				if(!M.contains(edge))
					newM.add(edge);
			}
			//Update M with newM
			M = new ArrayList<int[]>();
			int newMSize = newM.size(); 
			for(int i = 0; i < newMSize; ++i) {
				edge = newM.get(i);
				int[] newEdge = new int[2];
				newEdge[0] = edge[0];
				newEdge[1] = edge[1];
				M.add(newEdge);
			}
			
			//Build the array with costs 'd'
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
			
			for(int i = 0; i < hcmSize; ++i) {
				pLine[i]   += d[i + 1];
				pColumn[i] += d[i + 1 + hcmSize];
			}
		}
		
		//Eliminate negative cost arcs and arcs linked to artificial nodes
		postProcessPerfectMatching(costMatrix, modifiedHungarianCostMatrix, M, input);
		
		//Initialize the output vector with zero, indicating the quotations
		//are invalid. Then we assign the Perfect Bipartite Matching solution 
		//to the output
		int[] edge;
		int predictedOutputSize = predictedOutput.size();
		int mSize               = M.size();
		for(int i = 0; i < predictedOutputSize; ++i)
			predictedOutput.setAuthor(i, 0);
		for(int i = 0; i < mSize; ++i) {
			edge = M.get(i);
			//Convert M indexes to the Cost Matrix indexes, in which 'iIndex' represents 
			//quotations and 'jIndex' represents coreferences
			int iIndex = edge[0] - 1;
			int jIndex = edge[1] - 1 - hcmSize;
			
			predictedOutput.setAuthor(iIndex, jIndex);
		}
	}
	
	public double[][] generateCostMatrix(PQModel2 model, PQInput2 input, PQOutput2 correctOutput, double lossWeight) {
		ArrayList<Integer> coreferences = new ArrayList<Integer>();
		Quotation[] quotationIndexes    = input.getQuotationIndexes();
		
		// Create an ordered list of all the document coreferences. As there is no nested
		// coreference, keep only the coreference start token.
		for (int i = 0; i < quotationIndexes.length; ++i) {
			int numberOfCoreferences = quotationIndexes[i].getNumberOfCoreferences();
			
			for (int j = 0; j < numberOfCoreferences; ++j) {
				int[] coreferenceIndex = quotationIndexes[i].getCoreferenceIndex(j);
				int coreferenceStart   = coreferenceIndex[0];
				
				if (!coreferences.contains(coreferenceStart)) {
					// We search for a position to insert the new coreference in
					// the coreference list (O(n)). Then we insert the coreference in
					// specified position.
					int insertionIndex = Collections.binarySearch(coreferences, coreferenceStart);
					if (insertionIndex < 0)
						coreferences.add(-insertionIndex - 1, coreferenceStart);
					else
						coreferences.add(insertionIndex + 1, coreferenceStart);
				}
			}
		}
		
		int totalOfCoreferences = coreferences.size();
		double[][] costMatrix = new double[quotationIndexes.length][totalOfCoreferences];
		
		// Fill the cost matrix with zero.
		for (int i = 0; i < quotationIndexes.length; ++i)
			for (int j = 0; j < totalOfCoreferences; ++j)
				costMatrix[i][j] = 0d;
		
		// Generate all costs for the cost matrix. Lines correspond to 
		// quotations and columns correspond to coreferences.
		for (int i = 0; i < quotationIndexes.length; ++i) {
			int correctAuthor = 0;
			if(correctOutput != null)
				correctAuthor = correctOutput.getAuthor(i);
			
			int numberOfCoreferences = quotationIndexes[i].getNumberOfCoreferences();
			
			for (int j = 0; j < numberOfCoreferences; ++j) {
				int[] coreferenceIndex  = quotationIndexes[i].getCoreferenceIndex(j);
				int coreferencePosition = coreferences.indexOf(coreferenceIndex[0]);
				
				double currentLoss = 0d;
				if(correctOutput != null)
					currentLoss = (j != correctAuthor ? lossWeight : 0d);
				
				double cost = 0d;
				int featureIndex;
				Iterator<Integer> it = input.getFeatureCodes(i, j).iterator();
				while (it.hasNext()) {
					featureIndex = it.next();
					cost += currentLoss + model.getFeatureWeight(featureIndex);
				}
				
				costMatrix[i][coreferencePosition] = cost;
			}
		}
		
		return costMatrix;
	}
	
	public double[][] preProcessCostMatrix(double[][] costMatrix) {
		int lineSize   = costMatrix.length;
		int columnSize = (lineSize > 0 ? costMatrix[0].length : 0);
		int hcmSize    = (lineSize > columnSize ? lineSize : columnSize);
		
		double[][] hungarianCostMatrix = new double[hcmSize][hcmSize];
		double maxCost = -1d;
		
		//Fill artificial arcs with zero and eliminate negative costs
		for(int i = 0; i < hcmSize; ++i)
			for(int j = 0; j < hcmSize; ++j) {
				if(i < lineSize && j < columnSize) {
					if(costMatrix[i][j] > 0d)
						hungarianCostMatrix[i][j] = costMatrix[i][j];
					else
						hungarianCostMatrix[i][j] = 0d;
				}
				else
					hungarianCostMatrix[i][j] = 0d;
				
				if(maxCost < hungarianCostMatrix[i][j])
					maxCost = hungarianCostMatrix[i][j];
			}
		
		//Convert the cost matrix to be used by the Hungarian Algorithm
		for(int i = 0; i < hcmSize; ++i)
			for(int j = 0; j < hcmSize; ++j)
				hungarianCostMatrix[i][j] = maxCost - hungarianCostMatrix[i][j];
		
		return hungarianCostMatrix;
	}
	
	public void postProcessPerfectMatching(double[][] costMatrix, double[][] hungarianCostMatrix, ArrayList<int[]> M, PQInput2 input) {
		int[] edge;
		int lineSize                 = costMatrix.length;
		int columnSize               = (lineSize > 0 ? costMatrix[0].length : 0);
		int hcmSize                  = hungarianCostMatrix.length;
		int numberOfVertices         = hcmSize * 2 + 2;
		Quotation[] quotationIndexes = input.getQuotationIndexes();
		
		int mSize  = M.size();
		for(int i = mSize - 1; i >= 0; --i) {
			edge = M.get(i);
			
			//Take off M arcs that link some node to the artificial source or target
			if(edge[0] == 0 || edge[1] == numberOfVertices - 1) {
				M.remove(i);
				continue;
			}

			//Convert M indexes to the Cost Matrix indexes
			int iIndex = edge[0] - 1;
			int jIndex = edge[1] - 1 - hcmSize;
			
			//Eliminate arcs linked to artificial nodes
			if(iIndex >= lineSize)
				M.remove(i);
			else if(jIndex >= columnSize)
				M.remove(i);
			//Eliminate negative cost arcs
			else if(costMatrix[iIndex][jIndex] < 0d)
				M.remove(i);
			//Eliminate arcs linked to coreferences that cannot be associated with the
			//current quotation
			else if(jIndex >= quotationIndexes[iIndex].getNumberOfCoreferences())
				M.remove(i);
		}
	}

	@Override
	public void partialInference(Model model, ExampleInput input,
			ExampleOutput partiallyLabeledOutput, ExampleOutput predictedOutput) {
		// TODO Auto-generated method stub

	}

	@Override
	public void lossAugmentedInferenceWithPartiallyLabeledReference(
			Model model, ExampleInput input,
			ExampleOutput partiallyLabeledOutput,
			ExampleOutput referenceOutput, ExampleOutput predictedOutput,
			double lossAnnotatedWeight, double lossNonAnnotatedWeight) {
		// TODO Auto-generated method stub

	}

}
