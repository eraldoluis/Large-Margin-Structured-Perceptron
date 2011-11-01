package br.pucrio.inf.learn.structlearning.discriminative.application.pq.data;

import java.util.Collection;
import java.util.Iterator;

import br.pucrio.inf.learn.structlearning.discriminative.data.ExampleInput;

/**
 * Person-Quotation input structure.
 * 
 * @author eraldo
 * 
 */
public class PQInput2 implements ExampleInput {

	/**
	 * Identifier of the document that this example belongs to.
	 */
	private String docId;

	/**
	 * Index of this sequence input within the array of training examples, when
	 * it is a training example. Otherwise, this value is -1.
	 */
	private int trainingIndex;

	/**
	 * Quotation start and end indexes from the feed (example). 
	 */
	private int[][] quotationIndexes;
	
	/**
	 * Coreference start and end indexes from the feed (example).
	 */
	private int[][] corefIndexes;
	
	/**
	 * Sparse representation of binary features. This is an array of quotations.
	 * Each quotation is an array of coreferences. Each coreference is a list 
	 * of features describing the pair (quotation, coreference).
	 */
	private int[][][] features;

	/**
	 * Create a new PQ input using the given docID and the given list of
	 * feature codes.
	 * 
	 * @param docId
	 * @param quotations
	 * @param quotationIndexes
	 * @param corefIndexes
	 */
	public PQInput2(String docId,
			Collection<? extends Collection<? extends Collection<Integer>>> quotations,
			int[][] quotationIndexes,
			int[][] corefIndexes) {
		this.docId            = docId;
		this.quotationIndexes = quotationIndexes;
		this.corefIndexes     = corefIndexes;
		this.trainingIndex    = -1;

		// Array of features.
		this.features = new int[quotations.size()][][];
		
		int quotationIdx = 0;
		for (Collection<Collection<Integer>> quotation : quotations) {
			this.features[quotationIdx] = new int[quotation.size()][];

			int corefIdx = 0;
			for (Collection<Integer> coreference : quotation) {
				this.features[quotationIdx][corefIdx] = new int[coreference.size()];
				
				int ftrIdx = 0;
				for (int ftr : coreference) {
					this.features[quotationIdx][corefIdx][ftrIdx] = ftr;
	
					++ftrIdx;
				}
				
				++corefIdx;
			}

			++quotationIdx;
		}
	}
	
	/**
	 * Create a new PQ input using the given docID and the given list of
	 * feature codes.
	 * 
	 * @param docId
	 * @param trainingIndex
	 * @param tokens
	 */
	public PQInput2(String docId, int trainingIndex,
			Collection<? extends Collection<? extends Collection<Integer>>> quotations,
			int[][] quotationIndexes,
			int[][] corefIndexes) {
		this(docId, quotations, quotationIndexes, corefIndexes);
		this.trainingIndex = trainingIndex;
	}

	@Override
	public String getId() {
		return null;
	}

	@Override
	public PQOutput createOutput() {
		return new PQOutput();
	}

	@Override
	public void normalize(double norm) {
		// TODO only to normalize input vectors
	}

	@Override
	public void sortFeatureValues() {
		// TODO only to use kernel functions
	}

	@Override
	public int getTrainingIndex() {
		return -1;
	}
	
	public int size() {
		return features.length;
	}

	public Iterable<Integer> getFeatureCodes(int token) {
		return new FeatureCodeIterator(token);
	}

	private class FeatureCodeIterator implements Iterator<Integer>,
			Iterable<Integer> {

		/**
		 * Token index whose features this iterator iterates over.
		 */
		private int token;

		/**
		 * Current index within the feature array.
		 */
		private int curIndex;

		/**
		 * Create an iterator over the features of the given token.
		 * 
		 * @param token
		 */
		public FeatureCodeIterator(int token) {
			this.token = token;
			this.curIndex = -1;
		}

		@Override
		public boolean hasNext() {
			return curIndex < features[token].length - 1;
		}

		@Override
		public Integer next() {
			++curIndex;
			return features[token][curIndex];
		}

		@Override
		public void remove() {
			throw new UnsupportedOperationException(
					"This is an immutable iterator. One cannot remove an item.");
		}

		@Override
		public Iterator<Integer> iterator() {
			return this;
		}

	}

}
