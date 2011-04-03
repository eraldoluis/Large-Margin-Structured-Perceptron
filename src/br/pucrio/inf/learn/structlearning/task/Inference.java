package br.pucrio.inf.learn.structlearning.task;

import br.pucrio.inf.learn.structlearning.data.ExampleInput;
import br.pucrio.inf.learn.structlearning.data.ExampleOutput;

/**
 * Interface to task-specific procedures.
 * 
 * @author eraldof
 * 
 */
public interface Inference {

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

	/**
	 * Loss-augmented inference algorithm. Fill the given output structure using
	 * an augmented objective function that considers a weighted loss function
	 * through the margin rescaling approach.
	 * 
	 * @param model
	 *            the model containing the parameters necessary to the
	 *            inference.
	 * @param input
	 *            the input structure.
	 * @param referenceOutput
	 *            reference output structure that is used to calculate the loss
	 *            function.
	 * @param inferedOutput
	 *            the infered output structure that will be filled in this
	 *            method.
	 * @param lossWeight
	 *            the weight of the loss function in the objective function.
	 */
	void lossAugmentedInference(Model model, ExampleInput input,
			ExampleOutput referenceOutput, ExampleOutput inferedOutput,
			double lossWeight);

}
