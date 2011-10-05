package br.pucrio.inf.learn.structlearning.discriminative.application.pq.data;

import br.pucrio.inf.learn.structlearning.discriminative.data.ExampleInput;

/**
 * Person-Quotation input structure.
 * 
 * @author eraldo
 * 
 */
public class PQInput implements ExampleInput {

	/**
	 * Sparse representation of binary features. This is an array of quotations.
	 * Each quotation is an array of persons. Each person is a list of features
	 * * describing the pair (quotation,person).
	 */
	private int[][][] features;

	@Override
	public String getId() {
		return null;
	}

	@Override
	public PQOutput createOutput() {
		return new PQOutput(features.length);
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

}
