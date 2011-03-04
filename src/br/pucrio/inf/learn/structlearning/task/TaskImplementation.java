package br.pucrio.inf.learn.structlearning.task;

import br.pucrio.inf.learn.structlearning.data.ExampleInput;
import br.pucrio.inf.learn.structlearning.data.ExampleOutput;

/**
 * Interface to task-specific procedures.
 * 
 * @author eraldof
 * 
 */
public interface TaskImplementation {

	/**
	 * Fill the output structure using the given input and this model.
	 * 
	 * @param model
	 *            model to be used by the inference algorithm.
	 * @param input
	 *            the input structure.
	 * @param output
	 *            structure to store the predicted output, i.e., the output that
	 *            maximizes the objective function according to the given model.
	 */
	void inference(Model model, ExampleInput input, ExampleOutput output);

	/**
	 * Inference algorithm for partially-labeled example.
	 * 
	 * @param model
	 *            model to be used by the inference algorithm.
	 * @param input
	 *            the input structure.
	 * @param partiallyLabeledOutput
	 *            output structure containing the partial labeling.
	 * @param predictedOutput
	 *            structure to store the predicted output, i.e., the output that
	 *            maximizes the objective function according to the given model.
	 */
	void partialInference(Model model, ExampleInput input,
			ExampleOutput partiallyLabeledOutput, ExampleOutput predictedOutput);

}
