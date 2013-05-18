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
		// Find root (set id).
		int root = element;
		while (root != trees[root])
			root = trees[root];
		// Path compression.
		int q = element;
		while (q != trees[q]) {
			element = trees[q];
			trees[q] = root;
			q = element;
		}
		return root;
	}

	/**
	 * Merge <code>set1</code> and <code>set2</code>.
	 * 
	 * <b>Attention!</b> The given values <b>must</b> be set representants (the
	 * roots of theirs sets). Otherwise, this method can corrupt the data
	 * structure.
	 * 
	 * @param set1
	 * @param set2
	 */
	public void union(int set1, int set2) {
		trees[set2] = find(set1);
	}

	/**
	 * Merge the set of element <code>element1</code> and the set of element
	 * <code>element2</code>, if they are in different sets. Otherwise, do
	 * nothing.
	 * 
	 * @param element1
	 * @param element2
	 * @return <code>true</code> if the elements were in different sets,
	 *         <code>false</code> otherwise.
	 */
	public boolean unionElements(int element1, int element2) {
		int set1 = find(element1);
		int set2 = find(element2);
		if (set1 != set2) {
			union(set1, set2);
			return true;
		} else
			return false;
	}

	/**
	 * Cleat this partition so that it represents isolated elements.
	 */
	public void clear() {
		clear(trees.length);
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

	/**
	 * Return the number of elements in this object.
	 * 
	 * @return
	 */
	public int size() {
		return trees.length;
	}

}
