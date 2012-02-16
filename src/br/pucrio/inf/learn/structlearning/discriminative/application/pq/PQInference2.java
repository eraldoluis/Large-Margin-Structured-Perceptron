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

public class PQInference2 implements Inference {

	@Override
	public void inference(Model model, ExampleInput input, ExampleOutput output) {
		// TODO Auto-generated method stub
		inference((PQModel2) model, (PQInput2) input, (PQOutput2) output);
	}

	public void inference(PQModel2 model, PQInput2 input, PQOutput2 output) {
		// Generate candidates.
		Task[] tasks = generateWISCandidates(model, input, null, 0d);

		// Run WIS, which returns the indexes of the selected tasks.
		int[] solutionIndexes = wis(tasks);

		// Initialize the output vector with zero, indicating the quotations
		// are invalid. Then we assign the solution of WIS to the output.
		int outputSize = output.size();
		for(int i = 0; i < outputSize; ++i)
			output.setAuthor(i, 0);
		for(int i = 0; i < solutionIndexes.length; ++i)
			output.setAuthor(tasks[solutionIndexes[i]].getQuotationPosition(), tasks[solutionIndexes[i]].getCoreferencePosition());
	}

	@Override
	public void lossAugmentedInference(Model model, ExampleInput input,
			ExampleOutput referenceOutput, ExampleOutput predictedOutput,
			double lossWeight) {
		lossAugmentedInference((PQModel2) model, (PQInput2) input, (PQOutput2) referenceOutput,
							   (PQOutput2) predictedOutput, lossWeight);
	}

	public void lossAugmentedInference(PQModel2 model, PQInput2 input,
			PQOutput2 referenceOutput, PQOutput2 predictedOutput,
			double lossWeight) {
		// Generate candidates.
		Task[] tasks = generateWISCandidates(model, input, referenceOutput, lossWeight);
		
		// Run WIS, which returns the indexes of the selected tasks.
		int[] solutionIndexes = wis(tasks);

		// Initialize the output vector with -1, indicating the quotations
		// are invalid. Then we assign the WIS solution to the output.
		int predictedOutputSize = predictedOutput.size();
		for(int i = 0; i < predictedOutputSize; ++i)
			predictedOutput.setAuthor(i, 0);
		for(int i = 0; i < solutionIndexes.length; ++i)
			predictedOutput.setAuthor(tasks[solutionIndexes[i]].getQuotationPosition(), tasks[solutionIndexes[i]].getCoreferencePosition());
	}
	
	public void lossAugmentedInference(PQModel2 model, PQInput2 input,
			PQOutput2 referenceOutput, PQOutput2 predictedOutput,
			double lossWeight, boolean TOREMOVE) {
		double[][] costMatrix = generateCostMatrix(model, input, referenceOutput, lossWeight);
		double[][] hungarianCostMatrix = preProcessCostMatrix(costMatrix);
		
		int hcmSize = hungarianCostMatrix.length;
		
		double[] pLine   = new double[hcmSize];
		double[] pColumn = new double[hcmSize];
		
		//Initialize both p arrays
		for(int i = 0; i < hcmSize; ++i) {
			double minLine = Double.MAX_VALUE;
			for(int j = 0; j < hcmSize; ++j)
				if(hungarianCostMatrix[i][j] < minLine)
					minLine = hungarianCostMatrix[i][j];
			pLine[i]   = minLine;
			pColumn[i] = 0d;
		}
		
		ArrayList<int[]> M = new ArrayList<int[]>();
		while(!perfectMatching(M)) {
			//Update matrix costs
			for(int i = 0; i < hcmSize; ++i)
				for(int j = 0; j < hcmSize; ++j)
					hungarianCostMatrix[i][j] += pColumn[i] - pLine[i];
			
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
					graph.addEdge(i + 1, j + hcmSize + 1, hungarianCostMatrix[i][j]);
			
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
			//Update M with newM (if this copy does not work, use the commented code)
			M.clear();
			M = newM;
			//int newMSize = newM.size(); 
			//for(int i = 0; i < newMSize; ++i) {
			//	edge = newM.get(i);
			//	int[] newEdge = new int[2];
			//	newEdge[0] = edge[0];
			//	newEdge[1] = edge[1];
			//	M.add(newEdge);
			//}
			
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
			
			//TODO: Update both p arrays
			
		}
		
		
	}
	
	public double[][] generateAssignmentCandidates(PQModel2 model, PQInput2 input, PQOutput2 correctOutput, double lossWeight) {
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
		double[][] costs = new double[quotationIndexes.length][totalOfCoreferences];
		
		// Fill the cost matrix with zeros.
		for (int i = 0; i < quotationIndexes.length; ++i)
			for (int j = 0; j < totalOfCoreferences; ++j)
				costs[i][j] = 0d;
		
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
				
				costs[i][coreferencePosition] = cost;
			}
		}
		
		return costs;
	}

	public Task[] generateWISCandidates(PQModel2 model, PQInput2 input, PQOutput2 correctOutput, double lossWeight) {
		// Implementation of a method for comparison between two tasks.
		// It is used to find out in which position we have to insert the
		// new task in the list of tasks. Ordered by task end.
		Comparator<Task> comparator = new Comparator<Task>() {
			public int compare(Task t1, Task t2) {
				return Integer.valueOf(t1.getEnd()).compareTo(
						Integer.valueOf(t2.getEnd()));
			}
		};

		// Each quotation is associated to a number of coreferences, which are
		// the candidates to quotation author. We transform this problem into
		// the Weighted Interval Scheduling problem. The interval is:
		// [quotationStartToken, quotationEndToken]. The prize is given by the
		// sum of the feature weights that appears in the coreference.
		LinkedList<Task> tasks = new LinkedList<Task>();
		Quotation[] quotationIndexes = input.getQuotationIndexes();

		for (int i = 0; i < quotationIndexes.length; ++i) {
			int[] quotationIndex = quotationIndexes[i].getQuotationIndex();

			int correctAuthor = 0;
			if(correctOutput != null)
				correctAuthor = correctOutput.getAuthor(i);
			
			int numberOfCoreferences = quotationIndexes[i]
					.getNumberOfCoreferences();
			for (int j = 0; j < numberOfCoreferences; ++j) {
				int[] coreferenceIndex = quotationIndexes[i]
						.getCoreferenceIndex(j);


				int start = quotationIndex[0];
				int end   = quotationIndex[1];
				
				double currentLoss = 0d;
				if(correctOutput != null)
					currentLoss = (j != correctAuthor ? lossWeight : 0d);
				
				double prize = 0d;
				int featureIndex;
				Iterator<Integer> it = input.getFeatureCodes(i, j).iterator();
				while (it.hasNext()) {
					featureIndex = it.next();
					prize += currentLoss + model.getFeatureWeight(featureIndex);
				}

				Task task = new Task(start, end, prize, i, j, quotationIndex,
						coreferenceIndex);

				// We search for a position to insert the new task in
				// the task list (O(n)). Then we insert the task in
				// specified position.
				int insertionIndex = Collections.binarySearch(tasks, task,
						comparator);
				if (insertionIndex < 0)
					tasks.add(-insertionIndex - 1, task);
				else
					tasks.add(insertionIndex + 1, task);
			}
		}

		// Convert the LinkedList ''tasks'' in the array of
		// tasks ''tasksArray''.
		int tasksSize = tasks.size();
		Task[] tasksArray = new Task[tasksSize];
		for (int i = 0; i < tasksSize; ++i) {
			Task task = new Task(tasks.get(i));
			tasksArray[i] = task;
		}

		return tasksArray;
	}

	public int[] wis(Task[] tasks) {
		// Calculate the predecessors of each task.
		calculatePredecessors(tasks);

		// Store each subproblem in a cache. The boolean array
		// indicates if the indexed position is still empty.
		double[] cache = new double[tasks.length];
		boolean[] cacheEmpty = new boolean[tasks.length];

		for (int i = 0; i < cacheEmpty.length; ++i) {
			cacheEmpty[i] = true;
		}

		// Compute the optimal solution.
		computeOptimal(tasks, tasks.length - 1, cache, cacheEmpty);

		// Find the tasks that constitute the solution for the problem.
		ArrayList<Integer> solutionIndexes = new ArrayList<Integer>();
		findSolution(tasks, tasks.length - 1, cache, solutionIndexes);

		// Convert the ArrayList ''solutionIndexes'' in the array of
		// integers ''solutionIndexesArray''.
		int solutionIndexesSize = solutionIndexes.size();
		int[] solutionIndexesArray = new int[solutionIndexesSize];

		for (int i = 0; i < solutionIndexesSize; ++i)
			solutionIndexesArray[i] = solutionIndexes.get(i);

		return solutionIndexesArray;
	}

	public void findSolution(Task[] tasks, int index, double[] cache,
			ArrayList<Integer> solutionIndexes) {
		if (index >= 0) {
			double prize1 = tasks[index].getPrize();
			int predecessorIndex = tasks[index].getPredecessorIndex();
			if (predecessorIndex >= 0)
				prize1 += cache[predecessorIndex];

			double prize2 = (index > 0 ? cache[index - 1] : 0d);

			if (prize1 > prize2) {
				solutionIndexes.add(index);
				findSolution(tasks, tasks[index].getPredecessorIndex(), cache,
						solutionIndexes);
			} else
				findSolution(tasks, index - 1, cache, solutionIndexes);
		}
	}

	public double computeOptimal(Task[] tasks, int index, double[] cache,
			boolean[] cacheEmpty) {
		if (index == -1)
			return 0d;

		if (cacheEmpty[index]) {
			double prize1 = tasks[index].getPrize()
					+ computeOptimal(tasks, tasks[index].getPredecessorIndex(),
							cache, cacheEmpty);
			double prize2 = computeOptimal(tasks, index - 1, cache, cacheEmpty);

			cacheEmpty[index] = false;
			if (prize1 > prize2)
				cache[index] = prize1;
			else
				cache[index] = prize2;
		}

		return cache[index];
	}

	public void calculatePredecessors(Task[] tasks) {
		for (int i = 0; i < tasks.length; ++i) {
			Task task = tasks[i];

			for (int j = i; j >= 0; --j) {
				Task predecessorTask = tasks[j];

				if (task.getStart() > predecessorTask.getEnd()) {
					task.setPredecessorIndex(j);
					break;
				}
			}
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
