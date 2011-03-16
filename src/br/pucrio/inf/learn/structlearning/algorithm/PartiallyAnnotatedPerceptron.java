package br.pucrio.inf.learn.structlearning.algorithm;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;

import br.pucrio.inf.learn.structlearning.application.sequence.SequenceInput;
import br.pucrio.inf.learn.structlearning.application.sequence.SequenceOutput;
import br.pucrio.inf.learn.structlearning.data.ExampleInput;
import br.pucrio.inf.learn.structlearning.data.ExampleOutput;
import br.pucrio.inf.learn.structlearning.driver.TrainHmmMain;
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

			if (TrainHmmMain.print)
				System.out.println("#");

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

			if (TrainHmmMain.print && loss != 0d) {
				System.out.println("\nLoss: " + loss);
				SequenceInput seqIn = (SequenceInput) input;
				SequenceOutput seqCorOut = (SequenceOutput) correctOutput;
				SequenceOutput seqFilOut = (SequenceOutput) filledPartiallyLabeledOutput;
				SequenceOutput seqPreOut = (SequenceOutput) predictedOutput;
				for (int tkn = 0; tkn < seqIn.size(); ++tkn) {
					System.out.print(TrainHmmMain.featureEncoding
							.getValueByCode(seqIn.getFeature(tkn, 0))
							+ "_"
							+ TrainHmmMain.stateEncoding
									.getValueByCode(seqCorOut.getLabel(tkn))
							+ "_"
							+ TrainHmmMain.stateEncoding
									.getValueByCode(seqFilOut.getLabel(tkn))
							+ "_"
							+ TrainHmmMain.stateEncoding
									.getValueByCode(seqPreOut.getLabel(tkn))
							+ "  ");
				}
				System.out.println();
			}

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
