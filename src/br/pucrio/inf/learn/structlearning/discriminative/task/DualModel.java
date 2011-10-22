package br.pucrio.inf.learn.structlearning.discriminative.task;

import br.pucrio.inf.learn.structlearning.discriminative.data.ExampleOutput;

/**
 * Dual model for structured problems. It comprises a subset of the training
 * examples that are called support vectors (SV). In structured learning, a SV
 * usually is an example part, the fundamental part that forms examples. For
 * instance, in sentence labeling in text, the part is usually a token.
 * 
 * @author eraldo
 * 
 */
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

	/**
	 * Distill the set of support vectors, trying to reduce its size by removing
	 * redundant support vectors.
	 * 
	 * @param inference
	 * @param lossWeight
	 * @param outputsCache
	 */
	public void distill(Inference inference, double lossWeight,
			ExampleOutput[] outputsCache);

	/**
	 * Return the number of examples with some support vector.
	 * 
	 * @return
	 */
	public int getNumberOfExamplesWithSupportVector();

	/**
	 * Return the number of support vectors in this model.
	 * 
	 * @return
	 */
	public int getNumberOfSupportVectors();

}
