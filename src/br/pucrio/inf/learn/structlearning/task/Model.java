package br.pucrio.inf.learn.structlearning.task;

import br.pucrio.inf.learn.structlearning.data.ExampleInput;
import br.pucrio.inf.learn.structlearning.data.ExampleOutput;

/**
 * Interface for a model. One shall implement this interface for any task.
 * 
 * @author eraldof
 * 
 */
public interface Model {

	/**
	 * Fill the output structure using the given input and this model.
	 * 
	 * @param input
	 * @param output
	 */
	public void inference(ExampleInput input, ExampleOutput output);

	/**
	 * Update this model according to the two outputs (correct and predicted)
	 * for the given input.
	 * 
	 * @param input
	 * @param outputCorrect
	 * @param outputPredicted
	 * @param learningRate
	 */
	public void update(ExampleInput input, ExampleOutput outputCorrect,
			ExampleOutput outputPredicted, double learningRate);

}
