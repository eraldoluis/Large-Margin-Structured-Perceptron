package br.pucrio.inf.learn.structlearning.discriminative.application.dp;

import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.TreeSet;

import br.pucrio.inf.learn.structlearning.discriminative.application.dp.data.DPColumnDataset;
import br.pucrio.inf.learn.structlearning.discriminative.application.dp.data.DPInput;
import br.pucrio.inf.learn.util.HashCodeUtil;

/**
 * Represent a dependency parsing edge corpus (column format) by an inverted
 * index. That is, for each feature value, store the list of edges where this
 * feature occurs.
 * 
 * @author eraldo
 * 
 */
public class InvertedIndex {

	/**
	 * Direct index.
	 */
	private DPColumnDataset corpus;

	/**
	 * Inverted index.
	 */
	private HashMap<BasicFeature, TreeSet<Edge>> index;

	/**
	 * Auxiliary basic feature.
	 */
	final private BasicFeature auxBasicFeature = new BasicFeature(0, 0);

	/**
	 * Create an inverted index that represents the given corpus.
	 * 
	 * @param corpus
	 */
	public InvertedIndex(DPColumnDataset corpus) {
		this.corpus = corpus;
		initIndex();
	}

	/**
	 * Fill the inverted index with the underlying corpus.
	 */
	private void initIndex() {
		index = new HashMap<BasicFeature, TreeSet<Edge>>();
		DPInput[] inputs = corpus.getInputs();
		int numFtrs = corpus.getNumberOfFeatures();
		for (int example = 0; example < inputs.length; ++example) {
			DPInput input = inputs[example];
			int len = input.getNumberOfTokens();
			for (int head = 0; head < len; ++head) {
				for (int dependent = 0; dependent < len; ++dependent) {
					int[] vals = input.getFeatures(head, dependent);
					if (vals == null || vals.length == 0)
						continue;
					for (int feature = 0; feature < numFtrs; ++feature)
						put(example, head, dependent, feature, vals[feature]);
				}
			}
			if ((example + 1) % 100 == 0) {
				System.out.print(".");
				System.out.flush();
			}
		}
		System.out.println();
	}

	/**
	 * Return the underlying corpus.
	 * 
	 * @return
	 */
	public DPColumnDataset getCorpus() {
		return corpus;
	}

	/**
	 * Include the given example in the list of the given feature value.
	 * 
	 * @param example
	 * @param head
	 * @param dependent
	 * @param feature
	 * @param value
	 */
	public void put(int example, int head, int dependent, int feature, int value) {
		auxBasicFeature.value = value;
		auxBasicFeature.feature = feature;
		TreeSet<Edge> set = index.get(auxBasicFeature);
		if (set == null) {
			set = new TreeSet<Edge>();
			index.put(new BasicFeature(feature, value), set);
		}
		set.add(new Edge(example, head, dependent));
	}

	/**
	 * Return a collection of examples where the given features occur
	 * simultaneously.
	 * 
	 * @param features
	 * @param values
	 * @return
	 */
	@SuppressWarnings({ "rawtypes" })
	public Collection<Edge> getExamplesWithFeatures(int[] features, int[] values) {
		// Sort features by example list size.
		final TreeSet[] exampleSets = new TreeSet[features.length];
		ArrayList<Integer> sortedFeatures = new ArrayList<Integer>(
				features.length);
		for (int idx = 0; idx < features.length; ++idx) {
			// Index array.
			sortedFeatures.add(idx);
			auxBasicFeature.feature = features[idx];
			auxBasicFeature.value = values[idx];
			exampleSets[idx] = index.get(auxBasicFeature);
			if (exampleSets[idx] == null || exampleSets[idx].size() == 0)
				// Some feature never occurs in the corpus.
				return null;
		}

		Collections.sort(sortedFeatures, new Comparator<Integer>() {
			@Override
			public int compare(Integer idx1, Integer idx2) {
				int size1 = exampleSets[idx1].size();
				int size2 = exampleSets[idx2].size();
				if (size1 == size2)
					return 0;
				if (size1 < size2)
					return -1;
				return 1;
			}
		});

		return intersection(exampleSets, sortedFeatures);
	}

	// @SuppressWarnings({ "unchecked", "rawtypes" })
	// private TreeSet<Edge> intersection(TreeSet[] exampleSets,
	// ArrayList<Integer> sortedFeatures) {
	// TreeSet<Edge> examples = (TreeSet<Edge>) exampleSets[sortedFeatures
	// .get(0)].clone();
	// for (int idx = 1; idx < sortedFeatures.size(); ++idx) {
	// Iterator<Edge> it1 = examples.iterator();
	// Iterator<Edge> it2 = exampleSets[sortedFeatures.get(idx)]
	// .iterator();
	//
	// if (!it1.hasNext() || !it2.hasNext()) {
	// // Empty intersection.
	// examples = null;
	// break;
	// }
	//
	// // First elements.
	// Edge e1 = it1.next();
	// Edge e2 = it2.next();
	//
	// while (true) {
	// // Compare the two current elements.
	// int comp = e1.compareTo(e2);
	//
	// if (comp == 0) {
	// // List 1 current element is within list 2.
	// if (it1.hasNext())
	// e1 = it1.next();
	// else {
	// e1 = null;
	// break;
	// }
	//
	// if (it2.hasNext())
	// e2 = it2.next();
	// else
	// break;
	// } else if (comp < 0) {
	// // List 1 current element is not list 2.
	// it1.remove();
	// if (it1.hasNext())
	// e1 = it1.next();
	// else {
	// e1 = null;
	// break;
	// }
	// } else if (comp > 0) {
	// // Keep searching for list 1 current element in list 2.
	// if (it2.hasNext())
	// e2 = it2.next();
	// else
	// break;
	// }
	// }
	//
	// if (e1 != null)
	// it1.remove();
	//
	// // Remove all remaining elements from list 1.
	// while (it1.hasNext()) {
	// it1.next();
	// it1.remove();
	// }
	// }
	//
	// return examples;
	// }

	@SuppressWarnings({ "unchecked", "rawtypes" })
	private TreeSet<Edge> intersection(TreeSet[] exampleSets,
			ArrayList<Integer> sortedFeatures) {
		TreeSet<Edge> examples = (TreeSet<Edge>) exampleSets[sortedFeatures
				.get(0)].clone();
		for (int idx = 1; idx < sortedFeatures.size(); ++idx)
			examples.retainAll(exampleSets[sortedFeatures.get(idx)]);
		return examples;
	}

	/**
	 * Basic (not compound) feature. The inverted index is based on basic
	 * feature values.
	 * 
	 * @author eraldo
	 * 
	 */
	private static class BasicFeature {
		/**
		 * Feature code.
		 */
		public int feature;

		/**
		 * Feature value.
		 */
		public int value;

		/**
		 * Intialize a basic feature object.
		 * 
		 * @param feature
		 * @param value
		 */
		public BasicFeature(int feature, int value) {
			this.feature = feature;
			this.value = value;
		}

		@Override
		public int hashCode() {
			return HashCodeUtil.hash(feature, value);
		}

		@Override
		public boolean equals(Object obj) {
			if (!(obj instanceof BasicFeature))
				return false;
			BasicFeature other = (BasicFeature) obj;
			return feature == other.feature && value == other.value;
		}
	}

	/**
	 * Head-dependent identification along with its example index. For each
	 * basic feature value, the inverted index stores the list of edges with
	 * this value.
	 * 
	 * @author eraldo
	 * 
	 */
	public static class Edge implements Comparable<Edge> {
		/**
		 * Example index.
		 */
		public int example;

		/**
		 * Head token index.
		 */
		public int head;

		/**
		 * Dependent token index.
		 */
		public int dependent;

		/**
		 * Initialize an edge object.
		 * 
		 * @param example
		 * @param head
		 * @param dependent
		 */
		public Edge(int example, int head, int dependent) {
			this.example = example;
			this.head = head;
			this.dependent = dependent;
		}

		@Override
		public int hashCode() {
			return HashCodeUtil.hash(example,
					HashCodeUtil.hash(head, dependent));
		}

		@Override
		public boolean equals(Object obj) {
			if (!(obj instanceof Edge))
				return false;
			Edge other = (Edge) obj;
			return example == other.example && head == other.head
					&& dependent == other.dependent;
		}

		@Override
		public int compareTo(Edge o) {
			if (example < o.example)
				return -1;
			if (example > o.example)
				return 1;
			if (head < o.head)
				return -1;
			if (head > o.head)
				return 1;
			if (dependent < o.dependent)
				return -1;
			if (dependent > o.dependent)
				return 1;
			return 0;
		}
	}
}
