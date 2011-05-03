package br.pucrio.inf.learn.structlearning.algorithm;

import br.pucrio.inf.learn.structlearning.data.ExampleInput;
import br.pucrio.inf.learn.structlearning.data.ExampleOutput;
import br.pucrio.inf.learn.structlearning.data.StringEncoding;

/**
 * Interface for a structured learning algorithm.
 * 
 * @author eraldof
 * 
 */
public interface StructuredAlgorithm {

	/**
	 * Train the model with the given examples. Corresponding inputs and outputs
	 * must be in the same order.
	 * 
	 * @param inputs
	 * @param outputs
	 */
	public void train(ExampleInput[] inputs, ExampleOutput[] outputs,
			StringEncoding featureEncoding, StringEncoding stateEncoding);

	/**
	 * Train a model on two datasets. The first dataset (A) has a different
	 * weight and this weight is used to modify the sampling probability of the
	 * examples such that the probability of picking an example from the A is
	 * equal to weightA.
	 * 
	 * @param inputsA
	 * @param outputsA
	 * @param weightA
	 *            weight of the first dataset (A) between 0 and 1.
	 * @param weightStep
	 *            if this value is greater than zero, then starts with a weight
	 *            of 1 for the first dataset and after each epoch increase this
	 *            weight by this step value.
	 * @param inputsB
	 * @param outputsB
	 * @param featureEncoding
	 * @param stateEncoding
	 */
	public void train(ExampleInput[] inputsA, ExampleOutput[] outputsA,
			double weightA, double weightStep, ExampleInput[] inputsB,
			ExampleOutput[] outputsB, StringEncoding featureEncoding,
			StringEncoding stateEncoding);

}
