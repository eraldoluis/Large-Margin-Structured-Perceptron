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
	 * Return a temporary feature for the given token.
	 * 
	 * @param input
	 * @param idxHead
	 * @param idxDep
	 * @param idxTemplate
	 * @return
	 */
	public Feature getInstance(DPInput input, int idxHead, int idxDep,
			int idxTemplate);

	/**
	 * Instantiate a NEW feature for the given token.
	 * 
	 * @param input
	 * @param idxHead
	 * @param idxDep
	 * @param idxTemplate
	 * @return
	 */
	public Feature newInstance(DPInput input, int idxHead, int idxDep,
			int idxTemplate);

	/**
	 * Return the feature indexes used in this template.
	 * 
	 * @return
	 */
	public int[] getFeatures();

}
