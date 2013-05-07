package br.pucrio.inf.learn.structlearning.discriminative.application.rank;

import java.util.Collections;
import java.util.Comparator;
import java.util.Iterator;
import java.util.LinkedList;

import sun.reflect.generics.reflectiveObjects.NotImplementedException;

import br.pucrio.inf.learn.structlearning.discriminative.application.pq.data.PQInput2;
import br.pucrio.inf.learn.structlearning.discriminative.application.pq.data.PQOutput2;
import br.pucrio.inf.learn.structlearning.discriminative.application.pq.data.Quotation;
import br.pucrio.inf.learn.structlearning.discriminative.data.ExampleInput;
import br.pucrio.inf.learn.structlearning.discriminative.data.ExampleOutput;
import br.pucrio.inf.learn.structlearning.discriminative.task.Inference;
import br.pucrio.inf.learn.structlearning.discriminative.task.Model;

// Weighted Interval Scheduling
public class RankInference implements Inference {

	@Override
	public void inference(Model model, ExampleInput input, ExampleOutput output) {
		inference((PQModel2) model, (PQInput2) input, (PQOutput2) output);
	}

	@Override
	public void lossAugmentedInference(Model model, ExampleInput input,
			ExampleOutput referenceOutput, ExampleOutput predictedOutput,
			double lossWeight) {
		lossAugmentedInference((PQModel2) model, (PQInput2) input,
				(PQOutput2) referenceOutput, (PQOutput2) predictedOutput,
				lossWeight);
	}

	public void inference(PQModel2 model, PQInput2 input, PQOutput2 output) {
		// Generate candidates.
		Task[] tasks = generateWISCandidates(model, input, null, 0d);

		// Run WIS, which returns the indexes of the selected tasks.
		int[] solutionIndexes = WeightedIntervalScheduling
				.weightedIntervalScheduling(tasks);

		// Initialize the output vector with zero, indicating the quotations
		// are invalid. Then we assign the solution of WIS to the output.
		int outputSize = output.size();
		for (int i = 0; i < outputSize; ++i)
			output.setAuthor(i, 0);
		for (int i = 0; i < solutionIndexes.length; ++i)
			output.setAuthor(tasks[solutionIndexes[i]].getQuotationPosition(),
					tasks[solutionIndexes[i]].getCoreferencePosition());
	}

	public void lossAugmentedInference(PQModel2 model, PQInput2 input,
			PQOutput2 referenceOutput, PQOutput2 predictedOutput,
			double lossWeight) {
		// Generate candidates.
		Task[] tasks = generateWISCandidates(model, input, referenceOutput,
				lossWeight);

		// Run WIS, which returns the indexes of the selected tasks.
		int[] solutionIndexes = WeightedIntervalScheduling
				.weightedIntervalScheduling(tasks);

		// Initialize the output vector with zero, indicating the quotations
		// are invalid. Then we assign the WIS solution to the output.
		int predictedOutputSize = predictedOutput.size();
		for (int i = 0; i < predictedOutputSize; ++i)
			predictedOutput.setAuthor(i, 0);
		for (int i = 0; i < solutionIndexes.length; ++i)
			predictedOutput.setAuthor(
					tasks[solutionIndexes[i]].getQuotationPosition(),
					tasks[solutionIndexes[i]].getCoreferencePosition());
	}

	public Task[] generateWISCandidates(PQModel2 model, PQInput2 input,
			PQOutput2 correctOutput, double lossWeight) {
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
			if (correctOutput != null)
				correctAuthor = correctOutput.getAuthor(i);

			int numberOfCoreferences = quotationIndexes[i]
					.getNumberOfCoreferences();
			for (int j = 0; j < numberOfCoreferences; ++j) {
				int[] coreferenceIndex = quotationIndexes[i]
						.getCoreferenceIndex(j);

				int start = quotationIndex[0];
				int end = quotationIndex[1];

				double currentLoss = 0d;
				if (correctOutput != null)
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

	@Override
	public void partialInference(Model model, ExampleInput input,
			ExampleOutput partiallyLabeledOutput, ExampleOutput predictedOutput) {
		throw new NotImplementedException();
	}

	@Override
	public void lossAugmentedInferenceWithNonAnnotatedWeight(Model model,
			ExampleInput input, ExampleOutput partiallyLabeledOutput,
			ExampleOutput referenceOutput, ExampleOutput predictedOutput,
			double lossAnnotatedWeight, double lossNonAnnotatedWeight) {
		throw new NotImplementedException();
	}

}
