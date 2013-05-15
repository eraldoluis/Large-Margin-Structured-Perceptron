package br.pucrio.inf.learn.structlearning.discriminative.application.bisection;

import java.util.Collection;
import java.util.HashSet;
import java.util.Set;

import br.pucrio.inf.learn.structlearning.discriminative.data.ExampleOutput;

/**
 * Output structure for the bisection model. There are two representations in
 * this class, that are used exclusively, i.e., only one of them is active in
 * the same object. The first representation corresponds to binary relevance,
 * which is composed by two set of papers: confirmed and deleted (
 * <code>confirmedPapers</code> and <code>deletedPapers</code>). The second
 * representation corresponds to a rank of papers (<code>weightedPapers</code>).
 * The first representation is used to store gold-standard output structures.
 * The second is used to store predicted output structures.
 * 
 * @author eraldo
 * 
 */
public class BisectionOutput implements ExampleOutput {
	/**
	 * Ordered list of papers. Each element in this array is a paper index in
	 * the corresponding input structure. This array is used for predictions.
	 */
	public WeightedPaper[] weightedPapers;

	/**
	 * Set of confirmed papers. These should be ranked above of all deleted
	 * papers. This set is used for gold-standard outputs.
	 */
	private Set<Integer> confirmedPapers;

	/**
	 * Set of deleted papers. These should be ranked below of all confirmed
	 * papers. This set is used for gold-standard outputs.
	 */
	private Set<Integer> irrelevantItems;

	/**
	 * Size of this output structure, i.e., the number of candidate papers.
	 */
	private int size;

	/**
	 * Store the number of confirmed papers. Since there can be repeated
	 * confirmed papers, it is not right to calculate the number of them just by
	 * getting the size of the confirmed papers set.
	 */
	private int numConfirmedPapers;

	/**
	 * Create an output structure with the given size (number of papers). This
	 * structure is based on ranking, i.e., it is used for prediction.
	 * 
	 * @param size
	 */
	public BisectionOutput(int size) {
		this.size = size;
		this.weightedPapers = new WeightedPaper[size];
		for (int item = 0; item < size; ++item)
			weightedPapers[item] = new WeightedPaper(item, 0);
	}

	/**
	 * Create a golden output structure with the given two sets of confirmed and
	 * deleted papers.
	 * 
	 * @param confirmedPapers
	 * @param deletedPapers
	 */
	public BisectionOutput(Collection<Integer> confirmedPapers,
			Collection<Integer> deletedPapers) {
		this.numConfirmedPapers = confirmedPapers.size();
		this.size = numConfirmedPapers + deletedPapers.size();
		this.confirmedPapers = new HashSet<Integer>(confirmedPapers);
		this.irrelevantItems = new HashSet<Integer>(deletedPapers);
	}

	@Override
	public ExampleOutput createNewObject() {
		// Create a new output of the same size and based on ranking.
		return new BisectionOutput(size);
	}

	/**
	 * Return the size of this output structrue, i.e., the number of candidate
	 * papers within it.
	 * 
	 * @return
	 */
	public int size() {
		return size;
	}

	/**
	 * Return whether the given paper (index) is confirmed or not.
	 * 
	 * @param paper
	 * @return
	 */
	public boolean isConfirmed(int paper) {
		return confirmedPapers.contains(paper);
	}

	/**
	 * Return whether the given paper (index) is deleted or not.
	 * 
	 * @param paper
	 * @return
	 */
	public boolean isDeleted(int paper) {
		return irrelevantItems.contains(paper);
	}

	/**
	 * Compare the relevance of the given two candidate papers and return which
	 * from the two are more likely to be a confirmed paper. This operation
	 * works only for the binary relevance representation (gold standard).
	 * 
	 * @param paper1
	 * @param paper2
	 * @return 1 if paper1 is more relevant than paper2. -1 if paper2 is more
	 *         relevant than paper1. 0 if both papers have the same relevance.
	 */
	public int compare(int paper1, int paper2) {
		if (isConfirmed(paper1)) {
			if (isDeleted(paper2))
				return 1;
			else
				return 0;
		} else if (isDeleted(paper1)) {
			if (isConfirmed(paper2))
				return -1;
			else
				return 0;
		} else
			return 0;
	}

	/**
	 * Return the item in the given rank within this output ranking. This must
	 * be used only for predicted outputs.
	 * 
	 * @param rank
	 * @return
	 */
	public int getPaperAtIndex(int rank) {
		return weightedPapers[rank].paper;
	}

	/**
	 * Return the number of confirmed papers withint this structure. This method
	 * can be called only for gold-standard structures.
	 * 
	 * @return
	 */
	public int getNumberOfConfirmedPapers() {
		return numConfirmedPapers;
	}

	/**
	 * Return the number of deleted papers. This method can be called only for
	 * gold-standard structures.
	 * 
	 * @return
	 */
	public int getNumberOfDeletedPapers() {
		return size - numConfirmedPapers;
	}

	/**
	 * Represent a paper and its weight to be used to sort the list of papers
	 * according to the weights given by a model.
	 * 
	 * @author eraldo
	 * 
	 */
	public static class WeightedPaper implements Comparable<WeightedPaper> {
		/**
		 * Paper index as in the input structure.
		 */
		public int paper;

		/**
		 * Weight of the item.
		 */
		public double weight;

		/**
		 * Create an weighted item.
		 * 
		 * @param paper
		 * @param weight
		 */
		public WeightedPaper(int paper, double weight) {
			this.paper = paper;
			this.weight = weight;
		}

		@Override
		public int compareTo(WeightedPaper o) {
			if (weight > o.weight)
				return -1;
			if (weight < o.weight)
				return 1;
			return 0;
		}
	}

}
