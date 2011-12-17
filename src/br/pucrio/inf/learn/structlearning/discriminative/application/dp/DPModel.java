package br.pucrio.inf.learn.structlearning.discriminative.application.dp;

import br.pucrio.inf.learn.structlearning.discriminative.application.dp.data.DPInput;
import br.pucrio.inf.learn.structlearning.discriminative.task.Model;

/**
 * Basic methods required for a dependency parsing model.
 * 
 * @author eraldo
 * 
 */
public interface DPModel extends Model {

	/**
	 * Compute and return an edge score within the given input.
	 * 
	 * @param input
	 * @param idxHead
	 * @param idxDependent
	 * @return
	 */
	public double getEdgeScore(DPInput input, int idxHead, int idxDependent);

}
