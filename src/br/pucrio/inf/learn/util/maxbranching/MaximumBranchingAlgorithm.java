package br.pucrio.inf.learn.util.maxbranching;

/**
 * Maximum branching algorithm interface.
 * 
 * @author eraldo
 * 
 */
public interface MaximumBranchingAlgorithm {

	/**
	 * Find the maximum branching for the given graph. Fill the resulting
	 * branching in the given array of incoming edges.
	 * 
	 * @param numberOfNodes
	 * @param graph
	 * @param invertedMaxBranching
	 * @return
	 */
	public double findMaxBranching(int numberOfNodes, double[][] graph,
			int[] invertedMaxBranching);

	/**
	 * Realloc the internal data structures to handle graphs with the given
	 * number of nodes.
	 * 
	 * @param maxNumberOfNodes
	 */
	public void realloc(int maxNumberOfNodes);

	/**
	 * Activate or deactivate unique root checking.
	 * 
	 * @param check
	 */
	public void setCheckUniqueRoot(boolean check);

	/**
	 * Return whether unique root checking is activated or not.
	 * 
	 * @return
	 */
	public boolean isCheckUniqueRoot();

	/**
	 * Set whether to avoid negative-weight edges.
	 * 
	 * @param val
	 */
	public void setOnlyPositiveEdges(boolean val);

	/**
	 * Return whether only positive edges are allowed or not.
	 * 
	 * @return
	 */
	public boolean isOnlyPositiveEdges();

}
