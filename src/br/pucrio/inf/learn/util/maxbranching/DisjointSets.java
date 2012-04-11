package br.pucrio.inf.learn.util.maxbranching;


/**
 * Disjoint set forests with path compression heuristic. Elements and sets are
 * represented by integers in the interval {0, ..., n-1}, where n is the number
 * of possible elements. Each set is represented by a unique element number.
 * 
 * This implementation is based on Section 21.3 of the 3rd edition of Cormen,
 * Leiserson, Rivest and Stein's Introduction to Algorithms book (ignoring the
 * union by rank heuristic).
 * 
 * @author eraldo
 * 
 */
public class DisjointSets {

	/**
	 * Pointers to the parent of each element.
	 */
	private int[] trees;

	/**
	 * Create a copy of the given object.
	 * 
	 * @param copy
	 */
	public DisjointSets(DisjointSets copy) {
		this.trees = copy.trees.clone();
	}

	/**
	 * Create one set for each possible element.
	 * 
	 * @param numberOfElements
	 */
	public DisjointSets(int numberOfElements) {
		trees = new int[numberOfElements];
		for (int i = 0; i < numberOfElements; ++i)
			trees[i] = i;
	}

	/**
	 * Return the set where the given element is.
	 * 
	 * @param element
	 * @return
	 */
	public int find(int element) {
		if (trees[element] != element)
			trees[element] = find(trees[element]);
		return trees[element];
	}

	/**
	 * Include <code>set2</code> in <code>set1</code> and remove
	 * <code>set2</code>.
	 * 
	 * @param set1
	 * @param set2
	 */
	public void union(int set1, int set2) {
		trees[set2] = set1;
	}

	/**
	 * Clear this partition so that it represents isolated elements. The given
	 * <code>numberOfElements</code> is useful to use only a fraction of the
	 * elements.
	 * 
	 * @param numberOfElements
	 */
	public void clear(int numberOfElements) {
		for (int i = 0; i < numberOfElements; ++i)
			trees[i] = i;
	}

	@Override
	public DisjointSets clone() throws CloneNotSupportedException {
		return new DisjointSets(this);
	}

	/**
	 * Set this disjoint sets object with the same content of the given object.
	 * 
	 * @param copy
	 */
	public void setEqualTo(DisjointSets copy) {
		for (int idx = 0; idx < trees.length; ++idx)
			trees[idx] = copy.trees[idx];
	}

}
