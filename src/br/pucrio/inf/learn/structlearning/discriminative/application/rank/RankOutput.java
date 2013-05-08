package br.pucrio.inf.learn.structlearning.discriminative.application.rank;

import java.util.Collection;
import java.util.HashSet;
import java.util.Iterator;
import java.util.Set;

import br.pucrio.inf.learn.structlearning.discriminative.data.ExampleOutput;

/**
 * Output structure for ranking based on binary relevance or strict ordering. In
 * general, the binary relevance representation is used for golden output
 * structures and the strict ordering is used for predicted structures.
 * 
 * @author eraldo
 * 
 */
public class RankOutput implements ExampleOutput {
	/**
	 * Ordered list of items. Each element in this array is an item index in the
	 * corresponding input structure. This array is used for predictions.
	 */
	private int[] orderedItems;

	/**
	 * Set of relevant items. These should be ranked above of all irrelevant
	 * items. This set is used for golden outputs.
	 */
	private Set<Integer> relevantItems;

	/**
	 * Set of irrelevant items. These should be ranked below of all relevant
	 * items. This set is used for golden outputs.
	 */
	private Set<Integer> irrelevantItems;

	/**
	 * Size of this output structure, i.e., the number of its items.
	 */
	private int size;

	/**
	 * Auxiliary array to be used by the inference algorithm. Only for predicted
	 * outputs.
	 */
	public double[] weights;

	/**
	 * Create an output structure with the given size (number of items). This
	 * structure is based on strict ordering, i.e., it is used for prediction.
	 * 
	 * @param size
	 */
	public RankOutput(int size) {
		this.orderedItems = new int[size];
		this.weights = new double[size];
	}

	/**
	 * Create a prediction output structure with the given ordered list of
	 * items.
	 * 
	 * @param orderedItemsList
	 */
	public RankOutput(Collection<Integer> orderedItemsList) {
		int size = orderedItemsList.size();
		this.orderedItems = new int[size];
		Iterator<Integer> it = orderedItemsList.iterator();
		for (int idxItem = 0; idxItem < size; ++idxItem)
			orderedItems[idxItem] = it.next().intValue();
	}

	/**
	 * Create a golden output structure with the given two sets of relevant and
	 * irrelevant items.
	 * 
	 * @param relevantItems
	 * @param irrelevantItems
	 */
	public RankOutput(Collection<Integer> relevantItems,
			Collection<Integer> irrelevantItems) {
		this.size = relevantItems.size() + irrelevantItems.size();
		this.relevantItems = new HashSet<Integer>(relevantItems);
		this.irrelevantItems = new HashSet<Integer>(irrelevantItems);
	}

	@Override
	public ExampleOutput createNewObject() {
		// Create a new output of the same size and based on strict ordering.
		return new RankOutput(size);
	}

	/**
	 * Return the size of this output structrue, i.e., the number of items
	 * within it.
	 * 
	 * @return
	 */
	public int size() {
		return size;
	}

	/**
	 * Return whether the given item (index) is relevant or not.
	 * 
	 * @param item
	 * @return
	 */
	public boolean isRelevant(int item) {
		return relevantItems.contains(item);
	}

	/**
	 * Return whether the given item (index) is irrelevant or not.
	 * 
	 * @param item
	 * @return
	 */
	public boolean isIrrelevant(int item) {
		return irrelevantItems.contains(item);
	}

	/**
	 * Compare the relevance of the given two items (indexes) and return which
	 * from the two given items are more relevant. This operation works only for
	 * the binary relevance representation.
	 * 
	 * @param item1
	 * @param item2
	 * @return 1 if item1 is more relevant than item2. -1 if item2 is more
	 *         relevant than item1. 0 if both items have the same relevance.
	 */
	public int compare(int item1, int item2) {
		if (isRelevant(item1)) {
			if (isIrrelevant(item2))
				return 1;
			else
				return 0;
		} else if (isIrrelevant(item1)) {
			if (isRelevant(item2))
				return -1;
			else
				return 0;
		} else
			return 0;
	}

	/**
	 * Return the item in the given index of the ordered list of items (strict
	 * ordering). This must be used only for predicted outputs.
	 * 
	 * @param index
	 * @return
	 */
	public int getItemAtIndex(int index) {
		return orderedItems[index];
	}

	/**
	 * Return the number of relevant items. This method can be called only for
	 * golden structures.
	 * 
	 * @return
	 */
	public int getNumberOfRelevantItems() {
		return relevantItems.size();
	}

	/**
	 * Return the number of irrelevant items. This method can be called only for
	 * golden structure.
	 * 
	 * @return
	 */
	public int getNumberOfIrrelevantItems() {
		return irrelevantItems.size();
	}
}
