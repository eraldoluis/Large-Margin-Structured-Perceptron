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
		Task[] tasks = generateCandidates(model, input);

		// Run WIS, which returns the indexes of the selected tasks.
		int[] solutionIndexes = wis(tasks);

		// Initialize the output vector with -1, indicating the quotations
		// are invalid. Then we assign the solution of WIS to the output.
		int outputSize = output.size();
		for(int i = 0; i < outputSize; ++i)
			output.setAuthor(i, -1);
		for(int i = 0; i < solutionIndexes.length; ++i)
			output.setAuthor(tasks[solutionIndexes[i]].getQuotationPosition(), tasks[solutionIndexes[i]].getCoreferencePosition());
	}

	@Override
	public void lossAugmentedInference(Model model, ExampleInput input,
			ExampleOutput referenceOutput, ExampleOutput predictedOutput,
			double lossWeight) {
		inference((PQModel2) model, (PQInput2) input, (PQOutput2) predictedOutput);

	}

	public Task[] generateCandidates(PQModel2 model, PQInput2 input) {
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
		// [coreferenceStartToken, quotationEndToken], if the coreference
		// appears before the quotation; otherwise [quotationStartToken,
		// coreferenceEndToken]. The prize is given by the sum of the feature
		// weights that appears in the coreference.
		LinkedList<Task> tasks = new LinkedList<Task>();
		Quotation[] quotationIndexes = input.getQuotationIndexes();

		for (int i = 0; i < quotationIndexes.length; ++i) {
			int[] quotationIndex = quotationIndexes[i].getQuotationIndex();

			int numberOfCoreferences = quotationIndexes[i]
					.getNumberOfCoreferences();
			for (int j = 0; j < numberOfCoreferences; ++j) {
				int[] coreferenceIndex = quotationIndexes[i]
						.getCoreferenceIndex(j);

				int start;
				int end;
				if (quotationIndex[0] > coreferenceIndex[0]) {
					start = coreferenceIndex[0];
					end = quotationIndex[1];
				} else {
					start = quotationIndex[0];
					end = coreferenceIndex[1];
				}

				double prize = 0d;
				int featureIndex;
				Iterator<Integer> it = input.getFeatureCodes(i, j).iterator();
				while (it.hasNext()) {
					featureIndex = it.next();
					prize += model.getFeatureWeight(featureIndex);
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
	public void lossAugmentedInferenceWithNonAnnotatedWeight(
			Model model, ExampleInput input,
			ExampleOutput partiallyLabeledOutput,
			ExampleOutput referenceOutput, ExampleOutput predictedOutput,
			double lossAnnotatedWeight, double lossNonAnnotatedWeight) {
		// TODO Auto-generated method stub

	}

}
