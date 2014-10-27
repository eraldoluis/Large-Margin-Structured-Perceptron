package br.pucrio.inf.learn.structlearning.discriminative.algorithm;

import br.pucrio.inf.learn.structlearning.discriminative.data.ExampleInput;
import br.pucrio.inf.learn.structlearning.discriminative.data.ExampleInputArray;
import br.pucrio.inf.learn.structlearning.discriminative.data.ExampleOutput;
import br.pucrio.inf.learn.structlearning.discriminative.task.Model;

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
	public void train(ExampleInputArray inputs, ExampleOutput[] outputs);

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
	 */
	public void train(ExampleInputArray inputsA, ExampleOutput[] outputsA,
			double weightA, double weightStep, ExampleInputArray inputsB,
			ExampleOutput[] outputsB);

	/**
	 * Return the learned model. This method can only be called after a
	 * succesful training procedure.
	 * 
	 * @return
	 */
	public Model getModel();

	/**
	 * Indicate whether this algorithm shall consider partially annotated
	 * examples or not.
	 * 
	 * @param value
	 */
	public void setPartiallyAnnotatedExamples(boolean value);

	/**
	 * Set an object to listen the training process.
	 * 
	 * @param listener
	 */
	public void setListener(TrainingListener listener);

	/**
	 * Set the seed of the random-number generator. If this method is not
	 * called, the generator uses the default Java seed (a number very likely to
	 * be different from any other invocation).
	 * 
	 * @param seed
	 */
	public void setSeed(long seed);

}
