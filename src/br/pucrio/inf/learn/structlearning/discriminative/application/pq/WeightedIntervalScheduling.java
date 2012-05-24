package br.pucrio.inf.learn.structlearning.discriminative.application.pq;

import java.util.ArrayList;

public class WeightedIntervalScheduling {

	public static int[] weightedIntervalScheduling(Task[] tasks) {
		/*
		 * Find the tasks set that provides the greatest prize.
		 * 
		 * @param tasks: an array of tasks with start, end and prize.
		 * 
		 * @return: an array of integers which correspond to the
		 * 			task index in tasks array.
		 */
		
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

	public static void findSolution(Task[] tasks, int index, double[] cache,
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

	public static double computeOptimal(Task[] tasks, int index, double[] cache,
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

	public static void calculatePredecessors(Task[] tasks) {
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
}
