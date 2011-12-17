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
	 * Instantiate a feature for the given token.
	 * 
	 * @param input
	 * @param idxHead
	 * @param idxDep
	 * @param idxTemplate
	 * @return
	 */
	public Feature instantiate(DPInput input, int idxHead, int idxDep,
			int idxTemplate);

}
