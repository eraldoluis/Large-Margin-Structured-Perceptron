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
	 * Quotation index information. Each quotation has a list of coreference
	 * indexes, which are the candidates to be the quotation author.
	 */
	private Quotation[] quotationIndexes;

	/**
	 * Sparse representation of binary features. This is an array of quotations.
	 * Each quotation is an array of coreferences. Each coreference is a list of
	 * features describing the pair (quotation, coreference).
	 */
	private int[][][] features;

	/**
	 * Create a new PQ input using the given docID and the given list of feature
	 * codes.
	 * 
	 * @param docId
	 * @param quotations
	 * @param quotationIndexes
	 */
	public PQInput2(
			String docId,
			Collection<? extends Collection<? extends Collection<Integer>>> quotations,
			Collection<Quotation> quotationIndexes) {
		this.docId = docId;
		this.trainingIndex = -1;

		// Array of quotation indexes.
		this.quotationIndexes = new Quotation[quotationIndexes.size()];

		int quotationIdx = 0;
		for (Quotation quotation : quotationIndexes) {
			Quotation newQuotation = new Quotation(quotation);
			this.quotationIndexes[quotationIdx] = newQuotation;

			++quotationIdx;
		}

		// Array of features.
		this.features = new int[quotations.size()][][];

		quotationIdx = 0;
		for (Collection<? extends Collection<Integer>> quotation : quotations) {
			this.features[quotationIdx] = new int[quotation.size()][];

			int corefIdx = 0;
			for (Collection<Integer> coreference : quotation) {
				this.features[quotationIdx][corefIdx] = new int[coreference
						.size()];

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
	 * Create a new PQ input using the given docID and the given list of feature
	 * codes.
	 * 
	 * @param docId
	 * @param trainingIndex
	 * @param tokens
	 */
	public PQInput2(
			String docId,
			int trainingIndex,
			Collection<? extends Collection<? extends Collection<Integer>>> quotations,
			Collection<Quotation> quotationIndexes) {
		this(docId, quotations, quotationIndexes);
		this.trainingIndex = trainingIndex;
	}

	@Override
	public String getId() {
		return null;
	}

	@Override
	public PQOutput2 createOutput() {
		return new PQOutput2(features.length);
	}

	@Override
	public void normalize(double norm) {
		// TODO only to normalize input vectors
	}

	@Override
	public void sortFeatureValues() {
		// TODO only to use kernel functions
	}

	public String getDocId() {
		return docId;
	}

	@Override
	public int getTrainingIndex() {
		return -1;
	}

	public int getNumberOfQuotations() {
		return features.length;
	}
	
	public int getNumberOfCoreferences(int quotationIndex) {
		return features[quotationIndex].length;
	}
	
	public Quotation[] getQuotationIndexes() {
		return quotationIndexes;
	}
	
	public Iterable<Integer> getFeatureCodes(int quotationIndex, int coreferenceIndex) {
		return new FeatureCodeIterator(quotationIndex, coreferenceIndex);
	}

	
	/**
	 * Iterate over the features of a token.
	 * 
	 * @author eraldo
	 * 
	 */
	private class FeatureCodeIterator implements Iterator<Integer>,
			Iterable<Integer> {

		/**
		 * Quotation index whose features this iterator iterates over.
		 */
		private int quotationIndex;
		
		/**
		 * Coreference index whose features this iterator iterates over.
		 */
		private int coreferenceIndex;

		/**
		 * Current index within the feature array.
		 */
		private int curIndex;

		/**
		 * Create an iterator over the features of the given token.
		 * 
		 * @param token
		 */
		public FeatureCodeIterator(int quotationIndex, int coreferenceIndex) {
			this.quotationIndex = quotationIndex;
			this.coreferenceIndex = coreferenceIndex;
			this.curIndex = -1;
		}

		@Override
		public boolean hasNext() {
			return curIndex < features[quotationIndex][coreferenceIndex].length - 1;
		}

		@Override
		public Integer next() {
			++curIndex;
			return features[quotationIndex][coreferenceIndex][curIndex];
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
