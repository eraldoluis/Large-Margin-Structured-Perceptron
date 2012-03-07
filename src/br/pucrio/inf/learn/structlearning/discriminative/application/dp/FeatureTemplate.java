package br.pucrio.inf.learn.structlearning.discriminative.application.dp;

import br.pucrio.inf.learn.structlearning.discriminative.application.dp.data.DPInput;

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
	 * @param idxHead
	 * @param idxDep
	 * @return
	 */
	public Feature getInstance(DPInput input, int idxHead, int idxDep);

	/**
	 * Instantiate a NEW feature for the given token.
	 * 
	 * @param input
	 * @param idxHead
	 * @param idxDep
	 * @return
	 */
	public Feature newInstance(DPInput input, int idxHead, int idxDep);

}
