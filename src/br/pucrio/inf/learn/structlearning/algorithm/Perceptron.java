package br.pucrio.inf.learn.structlearning.algorithm;

import java.util.Random;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;

import br.pucrio.inf.learn.structlearning.application.sequence.SequenceInput;
import br.pucrio.inf.learn.structlearning.application.sequence.SequenceOutput;
import br.pucrio.inf.learn.structlearning.data.ExampleInput;
import br.pucrio.inf.learn.structlearning.data.ExampleOutput;
import br.pucrio.inf.learn.structlearning.data.StringEncoding;
import br.pucrio.inf.learn.structlearning.task.Inference;
import br.pucrio.inf.learn.structlearning.task.Model;
import br.pucrio.inf.learn.util.DebugUtil;

/**
 * Perceptron-trained linear classifier with structural examples. This class
 * implements the averaged version of the algorithm.
 * 
 * @author eraldof
 * 
 */
public class Perceptron {

	private static final Log LOG = LogFactory.getLog(Perceptron.class);

	/**
	 * Task-specific implementation of inference algorithms.
	 */
	protected Inference inferenceImpl;

	/**
	 * Task-specific model.
	 */
	protected Model model;

	/**
	 * Learning rate.
	 */
	protected double learningRate;

	/**
	 * Number of iterations.
	 */
	protected int numberOfEpochs;

	/**
	 * Last epoch executed.
	 */
	protected int epoch;

	/**
	 * This is the current iteration but counting one iteration for each
	 * example. This is necessary for the averaged-Perceptron implementation.
	 */
	protected int iteration;

	/**
	 * An object to observe the training process.
	 */
	protected Listener listener;

	/**
	 * Random-number generator.
	 */
	protected Random random;

	/**
	 * If this value is <code>false</code>, do not randomize the order to
	 * process the training examples. This value is <code>true</code> by
	 * default.
	 */
	protected boolean randomize;

	/**
	 * Report progress with this rate;
	 */
	protected double reportProgressRate;

	/**
	 * If <code>true</code> then consider partially-annotated examples.
	 */
	protected boolean partiallyAnnotatedExamples;

	/**
	 * Create a perceptron to train the given initial model using the default
	 * Collins' learning rate (1) and the default number of iterations (10).
	 * 
	 * @param initialModel
	 */
	public Perceptron(Inference taskImpl, Model initialModel) {
		this(taskImpl, initialModel, 10, 1d);
	}

	/**
	 * Create a perceptron to train the given initial model using the given
	 * number of iterations and learning rate.
	 * 
	 * @param initialModel
	 * @param numberOfEpochs
	 * @param learningRate
	 */
	public Perceptron(Inference taskImpl, Model initialModel,
			int numberOfEpochs, double learningRate) {
		this.inferenceImpl = taskImpl;
		this.model = initialModel;
		this.numberOfEpochs = numberOfEpochs;
		this.learningRate = learningRate;
		this.random = new Random();
		this.randomize = true;
	}

	/**
	 * Set to <code>true</code> if you want to consider partially-annotated
	 * examples.
	 * 
	 * @param value
	 */
	public void setPartiallyAnnotatedExamples(boolean value) {
		this.partiallyAnnotatedExamples = value;
	}

	public double getLearningRate() {
		return learningRate;
	}

	public void setLearningRate(double learningRate) {
		this.learningRate = learningRate;
	}

	public int getNumberOfEpochs() {
		return numberOfEpochs;
	}

	public void setNumberOfEpochs(int numberOfEpochs) {
		this.numberOfEpochs = numberOfEpochs;
	}

	public Model getModel() {
		return model;
	}

	/**
	 * Set the seed of the random-number generator. If this method is not
	 * called, the generator uses the default Java seed (a number very likely to
	 * be different from any other invocation).
	 * 
	 * @param seed
	 */
	public void setSeed(long seed) {
		random.setSeed(seed);
	}

	/**
	 * One can set this to <code>false</code> and avoid randomization of the
	 * order to process the training examples.
	 * 
	 * @param b
	 */
	public void setRandomize(boolean b) {
		randomize = b;
	}

	/**
	 * Set a listener object to observe the training process.
	 * 
	 * @param listener
	 */
	public void setListener(Listener listener) {
		this.listener = listener;
	}

	/**
	 * Train the model with the given examples. Corresponding inputs and outputs
	 * must be in the same order.
	 * 
	 * @param inputs
	 * @param outputs
	 */
	public void train(ExampleInput[] inputs, ExampleOutput[] outputs,
			StringEncoding featureEncoding, StringEncoding stateEncoding) {

		// Allocate predicted output objects for the training example.
		ExampleOutput[] predicteds = new ExampleOutput[outputs.length];
		for (int idx = 0; idx < inputs.length; ++idx)
			predicteds[idx] = inputs[idx].createOutput();

		if (listener != null)
			if (!listener.beforeTraining(inferenceImpl, model))
				return;

		iteration = 0;
		int epoch;
		for (epoch = 0; epoch < numberOfEpochs; ++epoch) {

			LOG.info("Perceptron epoch: " + epoch + "...");

			if (listener != null)
				if (!listener.beforeEpoch(inferenceImpl, model, epoch,
						iteration))
					// Stop training.
					break;

			// Train one epoch and get the accumulated loss.
			double loss = trainOneEpoch(inputs, outputs, predicteds,
					featureEncoding, stateEncoding);

			LOG.info("Training loss: " + loss);

			if (listener != null) {
				if (!listener.afterEpoch(inferenceImpl, model, epoch, loss,
						iteration)) {
					// Account the current epoch since it is concluded.
					++epoch;
					// Stop training.
					break;
				}
			}

		}

		if (listener != null)
			listener.afterTraining(inferenceImpl, model);

		// Averaged-Perceptron: average the final weights.
		model.average(iteration);
	}

	/**
	 * Train one epoch over the given input/output pairs.
	 * 
	 * @param inputs
	 *            list of input sequences
	 * @param outputs
	 *            list of correct output sequences
	 * @param predicteds
	 *            list of output sequences used to store the predicted values
	 * @param featureEncoding
	 *            encoding of feature values
	 * @param stateEncoding
	 *            encoding of state labels
	 * @return the sum of the losses over all examples through this epoch
	 */
	public double trainOneEpoch(ExampleInput[] inputs, ExampleOutput[] outputs,
			ExampleOutput[] predicteds, StringEncoding featureEncoding,
			StringEncoding stateEncoding) {

		// Accumulate the loss over all examples in this epoch.
		double loss = 0d;

		// Progress report.
		int reportProgressInterval = (int) (inputs.length * reportProgressRate);
		if (reportProgressInterval > 0)
			System.out.print("Progress: ");

		// Iterate over the training examples, updating the weight vector.
		for (int idx = 0; idx < inputs.length; ++idx, ++iteration) {

			int idxEx = idx;
			if (randomize)
				// Randomize the order to process the training examples.
				idxEx = random.nextInt(inputs.length);

			// Update the current model weights according with the predicted
			// output for this training example.
			loss += trainOneExample(inputs[idxEx], outputs[idxEx],
					predicteds[idxEx]);

			// Progress report.
			if (reportProgressInterval > 0
					&& (idx + 1) % reportProgressInterval == 0)
				System.out.print(Math.round((idx + 1) * 100d / inputs.length)
						+ "% ");

		}

		// Progress report.
		if (reportProgressInterval > 0)
			System.out.println("done.");

		return loss;

	}

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
			StringEncoding stateEncoding) {

		// Allocate predicted output objects for the training examples.
		ExampleOutput[] predictedsA = new ExampleOutput[outputsA.length];
		for (int idx = 0; idx < inputsA.length; ++idx)
			predictedsA[idx] = inputsA[idx].createOutput();
		ExampleOutput[] predictedsB = new ExampleOutput[outputsB.length];
		for (int idx = 0; idx < inputsB.length; ++idx)
			predictedsB[idx] = inputsB[idx].createOutput();

		if (listener != null)
			if (!listener.beforeTraining(inferenceImpl, model))
				return;

		iteration = 0;
		int epoch;
		for (epoch = 0; epoch < numberOfEpochs; ++epoch) {

			LOG.info("Perceptron epoch: " + epoch + "...");

			if (listener != null)
				if (!listener.beforeEpoch(inferenceImpl, model, epoch,
						iteration))
					// Stop training.
					break;

			// Adjust the weight for this epoch, if necessary.
			double epochWeightA = weightA;
			if (weightStep > 0d)
				epochWeightA = Math.max(weightA, 1d - epoch * weightStep);

			// Train one epoch and get the accumulated loss.
			double loss = trainOneEpoch(inputsA, outputsA, predictedsA,
					epochWeightA, inputsB, outputsB, predictedsB,
					featureEncoding, stateEncoding);

			LOG.info("Training loss: " + loss);

			if (listener != null) {
				if (!listener.afterEpoch(inferenceImpl, model, epoch, loss,
						iteration)) {
					// Account the current epoch since it is concluded.
					++epoch;
					// Stop training.
					break;
				}
			}

		}

		if (listener != null)
			listener.afterTraining(inferenceImpl, model);

		// Averaged-Perceptron: average the final weights.
		model.average(iteration);
	}

	/**
	 * Train one epoch over the two given datasets.
	 * 
	 * @param inputsA
	 * @param outputsA
	 * @param predictedsA
	 * @param weightA
	 * @param inputsB
	 * @param outputsB
	 * @param predictedsB
	 * @param featureEncoding
	 * @param stateEncoding
	 * @return
	 */
	public double trainOneEpoch(ExampleInput[] inputsA,
			ExampleOutput[] outputsA, ExampleOutput[] predictedsA,
			double weightA, ExampleInput[] inputsB, ExampleOutput[] outputsB,
			ExampleOutput[] predictedsB, StringEncoding featureEncoding,
			StringEncoding stateEncoding) {

		LOG.info("Weight of first dataset in this epoch: " + weightA);

		// Accumulate the loss over all examples in this epoch.
		double loss = 0d;

		// Use only the dataset A length to determine the length of an epoch.
		// This allows some easier comparison settings (per-epoch plots, for
		// instance) among models trained with different B datasets but only one
		// A dataset.
		int totalLength = inputsA.length;

		// Progress report.
		int reportProgressInterval = (int) (totalLength * reportProgressRate);
		if (reportProgressInterval > 0)
			System.out.print("Progress: ");

		// Iterate over the training examples, updating the weight vector.
		for (int idx = 0, idxA = 0, idxB = 0; idx < totalLength; ++idx, ++iteration) {

			// Randomly choose from A or B datasets.
			double aOrB = random.nextDouble();

			if (aOrB <= weightA) {

				// Train on A example.
				int idxEx = idxA++;

				if (randomize)
					// Randomize the order to process the training examples.
					idxEx = random.nextInt(inputsA.length);

				// Update the current model weights according with the predicted
				// output for this training example.
				loss += trainOneExample(inputsA[idxEx], outputsA[idxEx],
						predictedsA[idxEx]);

			} else {

				// Train on B example.
				int idxEx = idxB++;

				if (randomize)
					// Randomize the order to process the training examples.
					idxEx = random.nextInt(inputsB.length);

				// TODO debug
				// TrainHmmMain.print = true;

				// Update the current model weights according with the predicted
				// output for this training example.
				loss += trainOneExample(inputsB[idxEx], outputsB[idxEx],
						predictedsB[idxEx]);

				// TODO debug
				// TrainHmmMain.print = false;

			}

			// Progress report.
			if (reportProgressInterval > 0
					&& (idx + 1) % reportProgressInterval == 0)
				System.out.print(Math.round((idx + 1) * 100d / totalLength)
						+ "% ");

		}

		// Progress report.
		if (reportProgressInterval > 0)
			System.out.println("done.");

		return loss;

	}

	/**
	 * Update the current model using the two given outputs for one input.
	 * 
	 * @param input
	 * @param correctOutput
	 * @param predictedOutput
	 * @return the loss between the correct and the predicted outputs.
	 */
	public double trainOneExample(ExampleInput input,
			ExampleOutput correctOutput, ExampleOutput predictedOutput) {

		ExampleOutput referenceOutput = correctOutput;
		if (partiallyAnnotatedExamples) {
			// If the user asked to consider partially-labeled examples then
			// infer the missing values within the given correct output
			// structure before updating the current model.
			referenceOutput = correctOutput.createNewObject();
			inferenceImpl.partialInference(model, input, correctOutput,
					referenceOutput);
		}

		// Predict the best output with the current mobel.
		inferenceImpl.inference(model, input, predictedOutput);

		// Update the current model and return the loss for this example.
		double loss = model.update(input, referenceOutput, predictedOutput,
				learningRate);

		// TODO debug
		if (DebugUtil.print && loss != 0d)
			DebugUtil.printSequence((SequenceInput) input,
					(SequenceOutput) referenceOutput,
					(SequenceOutput) predictedOutput, loss);

		// Averaged-Perceptron: account the updates into the averaged weights.
		model.sumUpdates(iteration);

		return loss;
	}

	/**
	 * Set the rate in which the training progress (within each epoch) is
	 * reported.
	 * 
	 * @param rate
	 *            number between 0 and 1.
	 */
	public void setReportProgressRate(double rate) {
		reportProgressRate = rate;
	}

	/**
	 * Interface for listeners that observe the training algorithm.
	 * 
	 * @author eraldof
	 * 
	 */
	public interface Listener {

		/**
		 * Called before starting the training procedure.
		 * 
		 * @param impl
		 *            task-specific inference algorithms.
		 * @param curModel
		 *            the current model (no averaging).
		 * 
		 * @return <code>false</code> to not start the training procedure.
		 */
		boolean beforeTraining(Inference impl, Model curModel);

		/**
		 * Called after the training procedure ends.
		 * 
		 * @param impl
		 *            task-specific inference algorithms.
		 * @param curModel
		 *            the current model (no averaging).
		 */
		void afterTraining(Inference impl, Model curModel);

		/**
		 * Called before starting an epoch (processing the whole training set).
		 * 
		 * @param impl
		 *            task-specific inference algorithms.
		 * @param curModel
		 *            the current model (no averaging).
		 * @param epoch
		 *            the current epoch (starts in zero).
		 * @param iteration
		 *            current iteration (number of inference/update steps).
		 * 
		 * @return <code>false</code> to stop the training procedure.
		 */
		boolean beforeEpoch(Inference impl, Model curModel, int epoch,
				int iteration);

		/**
		 * Called after an epoch (processing the whole training set).
		 * 
		 * @param impl
		 *            task-specific inference algorithms.
		 * @param curModel
		 *            the current model (no averaging).
		 * @param epoch
		 *            the current epoch (starts in zero).
		 * @param loss
		 *            the training set loss during accumulated during the last
		 *            epoch.
		 * @param iteration
		 *            current iteration (number of inference/update steps).
		 * 
		 * @return
		 */
		boolean afterEpoch(Inference impl, Model curModel, int epoch,
				double loss, int iteration);

	}

}
