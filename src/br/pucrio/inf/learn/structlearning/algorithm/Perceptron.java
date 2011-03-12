package br.pucrio.inf.learn.structlearning.algorithm;

import java.util.Random;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;

import br.pucrio.inf.learn.structlearning.data.ExampleInput;
import br.pucrio.inf.learn.structlearning.data.ExampleOutput;
import br.pucrio.inf.learn.structlearning.data.StringEncoding;
import br.pucrio.inf.learn.structlearning.task.Model;
import br.pucrio.inf.learn.structlearning.task.TaskImplementation;

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
	protected TaskImplementation taskImpl;

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
	 * Report progress after processing this number of examples.
	 */
	protected int progressReportInterval;

	/**
	 * Create a perceptron to train the given initial model using the default
	 * Collins' learning rate (1) and the default number of iterations (10).
	 * 
	 * @param initialModel
	 */
	public Perceptron(TaskImplementation taskImpl, Model initialModel) {
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
	public Perceptron(TaskImplementation taskImpl, Model initialModel,
			int numberOfEpochs, double learningRate) {
		this.taskImpl = taskImpl;
		this.model = initialModel;
		this.numberOfEpochs = numberOfEpochs;
		this.learningRate = learningRate;
		this.random = new Random();
		this.randomize = true;
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
			if (!listener.beforeTraining(taskImpl, model))
				return;

		iteration = 0;
		int epoch;
		for (epoch = 0; epoch < numberOfEpochs; ++epoch) {

			LOG.info("Perceptron epoch: " + epoch + "...");

			if (listener != null)
				if (!listener.beforeEpoch(taskImpl, model, epoch, iteration))
					// Stop training.
					break;

			// Train one epoch and get the accumulated loss.
			double loss = trainOneEpoch(inputs, outputs, predicteds,
					featureEncoding, stateEncoding);

			LOG.info("Training loss: " + loss);

			if (listener != null) {
				if (!listener.afterEpoch(taskImpl, model, epoch, loss,
						iteration)) {
					// Account the current epoch since it is concluded.
					++epoch;
					// Stop training.
					break;
				}
			}

		}

		if (listener != null)
			listener.afterTraining(taskImpl, model);

		// Averaged-Perceptron: average the final weights.
		model.average(iteration);
	}

	public void train(ExampleInput[] inputsA, ExampleOutput[] outputsA,
			double weightA, ExampleInput[] inputsB, ExampleOutput[] outputsB,
			StringEncoding featureEncoding, StringEncoding stateEncoding) {

		// Allocate predicted output objects for the training example.
		ExampleOutput[] predictedsA = new ExampleOutput[outputsA.length];
		for (int idx = 0; idx < inputsA.length; ++idx)
			predictedsA[idx] = inputsA[idx].createOutput();
		ExampleOutput[] predictedsB = new ExampleOutput[outputsB.length];
		for (int idx = 0; idx < inputsB.length; ++idx)
			predictedsB[idx] = inputsB[idx].createOutput();

		if (listener != null)
			if (!listener.beforeTraining(taskImpl, model))
				return;

		iteration = 0;
		int epoch;
		for (epoch = 0; epoch < numberOfEpochs; ++epoch) {

			LOG.info("Perceptron epoch: " + epoch + "...");

			if (listener != null)
				if (!listener.beforeEpoch(taskImpl, model, epoch, iteration))
					// Stop training.
					break;

			// Train one epoch and get the accumulated loss.
			double loss = trainOneEpoch(inputsA, outputsA, predictedsA,
					weightA, inputsB, outputsB, predictedsB, featureEncoding,
					stateEncoding);

			LOG.info("Training loss: " + loss);

			if (listener != null) {
				if (!listener.afterEpoch(taskImpl, model, epoch, loss,
						iteration)) {
					// Account the current epoch since it is concluded.
					++epoch;
					// Stop training.
					break;
				}
			}

		}

		if (listener != null)
			listener.afterTraining(taskImpl, model);

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

		if (progressReportInterval > 0)
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

			if (progressReportInterval > 0
					&& (iteration + 1) % progressReportInterval == 0)
				System.out
						.print((100 * (iteration % inputs.length) / inputs.length)
								+ "% ");

		}

		if (progressReportInterval > 0)
			System.out.println("done.");

		return loss;

	}

	public double trainOneEpoch(ExampleInput[] inputsA,
			ExampleOutput[] outputsA, ExampleOutput[] predictedsA,
			double weightA, ExampleInput[] inputsB, ExampleOutput[] outputsB,
			ExampleOutput[] predictedsB, StringEncoding featureEncoding,
			StringEncoding stateEncoding) {

		// Accumulate the loss over all examples in this epoch.
		double loss = 0d;

		if (progressReportInterval > 0)
			System.out.print("Progress: ");

		int totalLength = inputsA.length + inputsB.length;

		// Iterate over the training examples, updating the weight vector.
		for (int idx = 0; idx < totalLength; ++idx, ++iteration) {

			// Randomize the order to process the training examples.
			double aOrB = random.nextDouble();
			if (aOrB <= weightA) {
				// Train on A example.
				int idxEx = random.nextInt(inputsA.length);
				// Update the current model weights according with the predicted
				// output for this training example.
				loss += trainOneExample(inputsA[idxEx], outputsA[idxEx],
						predictedsA[idxEx]);
			} else {
				// Train on B example.
				int idxEx = random.nextInt(inputsB.length);
				// Update the current model weights according with the predicted
				// output for this training example.
				loss += trainOneExample(inputsB[idxEx], outputsB[idxEx],
						predictedsB[idxEx]);
			}

			if (progressReportInterval > 0
					&& (iteration + 1) % progressReportInterval == 0)
				System.out
						.print((100 * (iteration % totalLength) / totalLength)
								+ "% ");

		}

		if (progressReportInterval > 0)
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
		// Predict the best output with the current mobel.
		taskImpl.inference(model, input, predictedOutput);

		// Update the current model and return the loss for this example.
		double loss = model.update(input, correctOutput, predictedOutput,
				learningRate);

		// Averaged-Perceptron: account the updates into the averaged weights.
		model.sumUpdates(iteration);

		return loss;
	}

	/**
	 * Set the interval in number of examples to report the training progress
	 * within each epoch. If this value is zero, no progress is reported.
	 * 
	 * @param progressReportInterval
	 */
	public void setProgressReportInterval(int progressReportInterval) {
		this.progressReportInterval = progressReportInterval;
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
		boolean beforeTraining(TaskImplementation impl, Model curModel);

		/**
		 * Called after the training procedure ends.
		 * 
		 * @param impl
		 *            task-specific inference algorithms.
		 * @param curModel
		 *            the current model (no averaging).
		 */
		void afterTraining(TaskImplementation impl, Model curModel);

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
		boolean beforeEpoch(TaskImplementation impl, Model curModel, int epoch,
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
		boolean afterEpoch(TaskImplementation impl, Model curModel, int epoch,
				double loss, int iteration);

	}

}
