package br.pucrio.inf.learn.structlearning.algorithm;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;

import br.pucrio.inf.learn.structlearning.data.ExampleInput;
import br.pucrio.inf.learn.structlearning.data.ExampleOutput;
import br.pucrio.inf.learn.structlearning.task.Model;
import br.pucrio.inf.learn.structlearning.task.TaskImplementation;

/**
 * Perceptron that considers partially-annotated examples. The examples
 * non-annotated tokens must be labeled with a special and non-valid state code,
 * i.e., a code less than zero or greater than the number of valid states.
 * 
 * @author eraldof
 * 
 */
public class PartiallyAnnotatedPerceptron extends Perceptron {

	private static final Log LOG = LogFactory
			.getLog(PartiallyAnnotatedPerceptron.class);

	public PartiallyAnnotatedPerceptron(TaskImplementation taskImpl,
			Model initialModel) {
		super(taskImpl, initialModel);
	}

	public PartiallyAnnotatedPerceptron(TaskImplementation taskImpl,
			Model initialModel, int numberOfIterations, double learningRate) {
		super(taskImpl, initialModel, numberOfIterations, learningRate);
	}

	@Override
	public double trainOneExample(ExampleInput input,
			ExampleOutput correctOutput, ExampleOutput predictedOutput) {

		try {

			ExampleOutput filledPartiallyLabeledOutput = (ExampleOutput) correctOutput
					.clone();

			// Label the non-annotated part of this example.
			taskImpl.partialInference(model, input, correctOutput,
					filledPartiallyLabeledOutput);

			// Predict the best output with the current mobel.
			taskImpl.inference(model, input, predictedOutput);

			// Update the current model and return the loss for this example.
			// TODO the loss should be insensible to the non-annotated part of
			// the given "correct" output.
			double loss = model.update(input, filledPartiallyLabeledOutput,
					predictedOutput, learningRate);

			// Averaged-Perceptron: account the updates into the averaged
			// weights.
			model.sumUpdates(iteration);

			return loss;

		} catch (CloneNotSupportedException e) {
			LOG.error("Cloning output structure", e);
			return 0d;
		}

	}
}
