package br.pucrio.inf.learn.structlearning.discriminative.task;

import br.pucrio.inf.learn.structlearning.discriminative.data.ExampleOutput;

public interface DualModel extends Model {

	/**
	 * Update model according to the difference between the given predicted
	 * output structure and the correct one.
	 * 
	 * @param sequenceId
	 * @param outputReference
	 * @param outputPredicted
	 * @param learnRate
	 * @return
	 */
	public double update(int sequenceId, ExampleOutput outputReference,
			ExampleOutput outputPredicted, double learnRate);

}
