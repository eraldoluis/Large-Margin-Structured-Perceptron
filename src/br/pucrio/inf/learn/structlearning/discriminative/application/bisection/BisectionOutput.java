package br.pucrio.inf.learn.structlearning.discriminative.application.bisection;

import java.util.Arrays;
import java.util.HashSet;
import java.util.Set;

import br.pucrio.inf.learn.structlearning.discriminative.data.ExampleOutput;
import br.pucrio.inf.learn.util.maxbranching.DisjointSets;
import br.pucrio.inf.learn.util.maxbranching.SimpleWeightedEdge;

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
	 * Flags indicating which papers are confirmed and which are deleted.
	 * 
	 * This structure is mainly used for gold-standard outputs. However,
	 * predicted outputs also use this structure to store the predicted split
	 * before sorting the papers within each split.
	 */
	private boolean[] confirmedPapers;

	/**
	 * Ordered list of papers. Each element in this array is a paper index in
	 * the corresponding input structure.
	 * 
	 * This array is used for predicted outputs. It corresponds to the final
	 * output, i.e., the ranked papers.
	 */
	public WeightedPaper[] weightedPapers;

	/**
	 * Edges of the maximum spaning tree (MST).
	 * 
	 * This list is used for predicted outputs to split the candidate papers in
	 * two connected componenets: confirmed and deleted.
	 */
	private Set<SimpleWeightedEdge> mst;

	/**
	 * Union-find structure to represent the nodes partition built by the MST
	 * algorithm.
	 */
	private DisjointSets partition;

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
		for (int idx = 0; idx < size; ++idx)
			weightedPapers[idx] = new WeightedPaper(idx, false, 0);
		this.mst = new HashSet<SimpleWeightedEdge>();
		this.confirmedPapers = new boolean[size];
		this.numberOfConfirmedPapers = -1;
		this.partition = new DisjointSets(size);
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
		this.weightedPapers = null;
		/*
		 * new WeightedPaper[size]; for (int paper = 0; paper < size; ++paper)
		 * weightedPapers[paper] = new WeightedPaper(paper, confirmed[paper],
		 * 0);
		 */
		this.mst = new HashSet<SimpleWeightedEdge>();
		this.confirmedPapers = confirmed;
		this.numberOfConfirmedPapers = 0;
		for (boolean b : confirmedPapers)
			if (b)
				++numberOfConfirmedPapers;
		this.partition = new DisjointSets(size);
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
	 * Return the list of edges comprising the predicted MST.
	 * 
	 * @return
	 */
	public Set<SimpleWeightedEdge> getMst() {
		return mst;
	}

	/**
	 * Return the number of confirmed papers for this author.
	 * 
	 * @return
	 */
	public int getNumberOfConfirmedPapers() {
		return numberOfConfirmedPapers;
	}

	/**
	 * Compute confirmed and deleted papers from the underlying MST.
	 */
	public void computeSplitFromMstPartition() {
		// Initially, consider every paper as confirmed.
		Arrays.fill(confirmedPapers, true);

		// Unflag deleted papers (all papers in the artificial node cluster).
		int delCluster = partition.find(0);
		for (int paper = 0; paper < size; ++paper) {
			if (partition.find(paper) == delCluster)
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
	 * Return the union-find structure used to run Kruskal algorithm.
	 * 
	 * @return
	 */
	public DisjointSets getPartition() {
		return partition;
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
	public static class WeightedPaper {
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
	}

}
