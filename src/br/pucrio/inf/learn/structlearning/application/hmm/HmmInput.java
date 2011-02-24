package br.pucrio.inf.learn.structlearning.application.hmm;

import java.util.Vector;

import br.pucrio.inf.learn.structlearning.data.ExampleInput;

/**
 * Sequence of tokens along their features.
 * 
 * @author eraldo
 * 
 */
public class HmmInput implements ExampleInput {

	private int numberOfFeatures;

	private Vector<int[]> tokens;

	public int size() {
		return tokens.size();
	}

	public int getNumberOfFeatures() {
		return numberOfFeatures;
	}

	public int getFeatureValue(int token, int feature) {
		return tokens.get(token)[feature];
	}

	public void setFeatureValue(int token, int feature, int value) {
		tokens.get(token)[feature] = value;
	}

}
