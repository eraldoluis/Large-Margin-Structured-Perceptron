package br.pucrio.inf.learn.structlearning.discriminative.application.dp;

import br.pucrio.inf.learn.structlearning.discriminative.data.ExampleInput;

/**
 * Represents a combination of basic features.
 * 
 * @author eraldo
 * 
 */
public interface FeatureTemplate {

	/**
	 * Return this template index within the template set.
	 * 
	 * @return
	 */
	public int getIndex();

	/**
	 * Return the feature indexes used in this template.
	 * 
	 * @return
	 */
	public int[] getFeatures();

	/**
	 * Return a temporary feature for the given token.
	 * 
	 * @param input
	 * @param params
	 * @return
	 */
	public Feature getInstance(ExampleInput input, Object... params);

	/**
	 * Instantiate a NEW feature for the given token.
	 * 
	 * @param input
	 * @param params
	 * @return
	 */
	public Feature newInstance(ExampleInput input, Object... params);

}
