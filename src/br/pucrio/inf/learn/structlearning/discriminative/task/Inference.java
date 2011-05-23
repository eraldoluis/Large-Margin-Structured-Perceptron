package br.pucrio.inf.learn.structlearning.discriminative.task;

import br.pucrio.inf.learn.structlearning.discriminative.data.ExampleInput;
import br.pucrio.inf.learn.structlearning.discriminative.data.ExampleOutput;

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

	/**
	 * Loss-augmented inference algorithm that condiders partially-annotated
	 * examples. The loss function has different weights depending on whether
	 * the considered elements is annotated or not.
	 * 
	 * @param model
	 *            the model with the feature weights.
	 * @param input
	 *            the input structure.
	 * @param lossPartiallyLabeledOutput
	 *            the partial output structure that is used to check whether the
	 *            element is annotated or not.
	 * @param referenceOutput
	 *            the reference output structure that is used to calculate the
	 *            loss function.
	 * @param inferedOutput
	 *            the infered output structure.
	 * @param lossAnnotatedWeight
	 *            weight used for missing elements that are annotated.
	 * @param lossNonAnnotatedWeight
	 *            weight used for missing elements that are not annotated.
	 */
	void lossAugmentedInference(Model model, ExampleInput input,
			ExampleOutput lossPartiallyLabeledOutput,
			ExampleOutput referenceOutput, ExampleOutput inferedOutput,
			double lossAnnotatedWeight, double lossNonAnnotatedWeight);

}
