package br.pucrio.inf.learn.structlearning.discriminative.application.bisection;

import java.util.Arrays;

import br.pucrio.inf.learn.structlearning.discriminative.data.ExampleOutput;
import br.pucrio.inf.learn.util.maxbranching.DisjointSets;

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
	 * the corresponding input structure. This array is used for predicted
	 * outputs.
	 */
	public WeightedPaper[] weightedPapers;

	/**
	 * Array of incident edges of the maximum spaning tree (MST). This structure
	 * is used only for predicted outputs.
	 */
	private int[] mst;

	/**
	 * Flags indicating which papers are confirmed and which are deleted. This
	 * structure is used only for gold-standard outputs.
	 */
	private boolean[] confirmedPapers;

	/**
	 * Size of this output structure, i.e., the number of candidate papers.
	 */
	private int size;

	/**
	 * Number of confirmed papers out of all candidate papers for this author.
	 */
	private int numberOfConfirmedPapers;

	/**
	 * Create an output structure with the given size (number of papers). This
	 * structure is based on ranking, i.e., it is used for prediction.
	 * 
	 * @param size
	 */
	public BisectionOutput(int size) {
		this.size = size;
		this.weightedPapers = new WeightedPaper[size];
		for (int paper = 0; paper < size; ++paper)
			weightedPapers[paper] = new WeightedPaper(paper, false, 0);
		this.mst = new int[size];
		this.confirmedPapers = new boolean[size];
		this.numberOfConfirmedPapers = -1;
	}

	/**
	 * Create a golden output structure with the given two sets of confirmed and
	 * deleted papers.
	 * 
	 * @param confirmedPapers
	 * @param deletedPapers
	 */
	public BisectionOutput(boolean[] confirmed) {
		this.size = confirmed.length;
		this.weightedPapers = new WeightedPaper[size];
		for (int paper = 0; paper < size; ++paper)
			weightedPapers[paper] = new WeightedPaper(paper, confirmed[paper],
					0);
		this.mst = new int[size];
		this.confirmedPapers = confirmed;
		this.numberOfConfirmedPapers = 0;
		for (boolean b : confirmedPapers)
			if (b)
				++numberOfConfirmedPapers;
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
		return confirmedPapers[paper];
	}

	/**
	 * Return whether the given paper (index) is deleted or not.
	 * 
	 * @param paper
	 * @return
	 */
	public boolean isDeleted(int paper) {
		return !confirmedPapers[paper];
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
	 * Return the array that represents a MST solution. This array stores, for
	 * each paper, the adjacent paper in the oriented tree.
	 * 
	 * @return
	 */
	public int[] getMst() {
		return mst;
	}

	public int getNumberOfConfirmedPapers() {
		return numberOfConfirmedPapers;
	}

	/**
	 * Compute confirmed and deleted papers from the underlying MST.
	 */
	public void computeSplitFromMst() {
		// Initially, every paper is confirmed.
		Arrays.fill(confirmedPapers, true);
		// Find clusters.
		DisjointSets clustering = new DisjointSets(size);
		for (int paper1 = 0; paper1 < size; ++paper1) {
			int paper2 = mst[paper1];
			if (paper2 < 0)
				continue;
			int cluster1 = clustering.find(paper1);
			int cluster2 = clustering.find(paper2);
			if (cluster1 != cluster2)
				clustering.union(cluster1, cluster2);
		}

		// Get deleted papers (papers connected to the artificial node).
		int delCluster = clustering.find(0);
		confirmedPapers[0] = false;
		for (int paper = 0; paper < size; ++paper) {
			if (clustering.find(paper) == delCluster)
				confirmedPapers[paper] = false;
		}
	}

	/**
	 * Copy the confirmed papers from the given output to this output.
	 * 
	 * @param correct
	 */
	public void setConfirmedPapersEqualTo(BisectionOutput correct) {
		for (int idx = 0; idx < size; ++idx)
			confirmedPapers[idx] = correct.confirmedPapers[idx];
	}

	/**
	 * Represent a paper along with its confirmed flag and its weight. Confirmed
	 * papers are ranked before not confirmed ones. Whenever two papers have the
	 * same confirmed flag, they are ranked based on their weights (higher
	 * weights are ranked first).
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
		 * Indicate whether this paper is confirmed or deleted.
		 */
		public boolean confirmed;

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
		public WeightedPaper(int paper, boolean confirmed, double weight) {
			this.paper = paper;
			this.confirmed = confirmed;
			this.weight = weight;
		}

		@Override
		public int compareTo(WeightedPaper o) {
			if (confirmed && !o.confirmed)
				return -1;
			if (!confirmed && o.confirmed)
				return 1;
			if (weight > o.weight)
				return -1;
			if (weight < o.weight)
				return 1;
			return 0;
		}
	}

}
