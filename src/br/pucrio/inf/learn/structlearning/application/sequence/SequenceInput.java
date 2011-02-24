package br.pucrio.inf.learn.structlearning.application.sequence;

import br.pucrio.inf.learn.structlearning.data.ExampleInput;

/**
 * Sequence of tokens along their features.
 * 
 * @author eraldo
 * 
 */
public interface SequenceInput extends ExampleInput {

	public int size();

	public int getNumberOfFeatures();

	public int getFeatureValue(int token, int feature);

	public void setFeatureValue(int token, int feature, int value);

}
