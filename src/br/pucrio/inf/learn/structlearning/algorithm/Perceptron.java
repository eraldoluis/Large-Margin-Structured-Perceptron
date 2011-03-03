package br.pucrio.inf.learn.structlearning.algorithm;

import java.util.Random;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;

import br.pucrio.inf.learn.structlearning.data.ExampleInput;
import br.pucrio.inf.learn.structlearning.data.ExampleOutput;
import br.pucrio.inf.learn.structlearning.data.StringEncoding;
import br.pucrio.inf.learn.structlearning.task.Model;

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

	protected boolean randomize;

	/**
	 * Create a perceptron to train the given initial model using the default
	 * Collins' learning rate (1) and the default number of iterations (10).
	 * 
	 * @param initialModel
	 */
	public Perceptron(Model initialModel) {
		this(initialModel, 10, 1d);
	}

	/**
	 * Create a perceptron to train the given initial model using the given
	 * number of iterations and learning rate.
	 * 
	 * @param initialModel
	 * @param numberOfIterations
	 * @param learningRate
	 */
	public Perceptron(Model initialModel, int numberOfIterations,
			double learningRate) {
		this.model = initialModel;
		this.numberOfEpochs = numberOfIterations;
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
	 * Train the model with the given examples, iterating according to the
	 * number of iterations. Corresponding inputs and outputs must be in the
	 * same order.
	 * 
	 * @param inputs
	 * @param outputs
	 */
	public void train(ExampleInput[] inputs, ExampleOutput[] outputs,
			StringEncoding featureEncoding, StringEncoding stateEncoding) {
		// Create a predicted output object for each example.
		ExampleOutput[] predicteds = new ExampleOutput[outputs.length];
		for (int idx = 0; idx < inputs.length; ++idx)
			predicteds[idx] = inputs[idx].createOutput();

		if (listener != null)
			if (!listener.beforeTraining(model))
				return;

		iteration = 0;
		int epoch;
		for (epoch = 0; epoch < numberOfEpochs; ++epoch) {

			LOG.info("Perceptron epoch: " + epoch + "...");

			if (listener != null)
				if (!listener.beforeEpoch(model, epoch, iteration))
					// Stop training.
					break;

			// Loss accumulated over all training examples.
			double loss = train(inputs, outputs, predicteds, featureEncoding,
					stateEncoding);

			LOG.info("Training loss: " + loss);

			if (listener != null) {
				if (!listener.afterEpoch(model, epoch, loss, iteration)) {
					// Account the current epoch since it was concluded.
					++epoch;
					// Stop training.
					break;
				}
			}

		}

		if (listener != null)
			listener.afterTraining(model);

		// Averaged-Perceptron: average the final weights.
		model.average(iteration);
	}

	/**
	 * Train one epoch over the given input/output/predicted triples.
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
	public double train(ExampleInput[] inputs, ExampleOutput[] outputs,
			ExampleOutput[] predicteds, StringEncoding featureEncoding,
			StringEncoding stateEncoding) {
		double loss = 0d;
		// Iterate over the training examples, updating the weight vector.
		for (int idx = 0; idx < inputs.length; ++idx, ++iteration) {
			int idxEx = idx;
			if (randomize)
				idxEx = random.nextInt(inputs.length);
			// Update the current model weights according with the predicted
			// output for this training example.
			loss += train(inputs[idxEx], outputs[idxEx], predicteds[idxEx]);
		}
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
	public double train(ExampleInput input, ExampleOutput correctOutput,
			ExampleOutput predictedOutput) {
		// Predict the best output with the current mobel.
		model.inference(input, predictedOutput);

		// Update the current model and return the loss for this example.
		double loss = model.update(input, correctOutput, predictedOutput,
				learningRate);

		// Averaged-Perceptron: account the updates into the averaged weights.
		model.sumUpdates(iteration);

		return loss;
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
		 * @param curModel
		 * @return <code>false</code> to not start the training procedure.
		 */
		boolean beforeTraining(Model curModel);

		/**
		 * Called after the training procedure ends.
		 * 
		 * @param curModel
		 */
		void afterTraining(Model curModel);

		/**
		 * Called before starting an epoch (processing the whole training set).
		 * 
		 * @param curModel
		 *            the current model (no averaging).
		 * @param epoch
		 *            the current epoch (starts in zero).
		 * @param iteration
		 *            current iteration (number of inference/update steps).
		 * @return <code>false</code> to stop the training procedure.
		 */
		boolean beforeEpoch(Model curModel, int epoch, int iteration);

		/**
		 * Called after an epoch (processing the whole training set).
		 * 
		 * @param curModel
		 *            the current model (no averaging).
		 * @param epoch
		 *            the current epoch (starts in zero).
		 * @param loss
		 *            the training set loss during accumulated during the last
		 *            epoch.
		 * @param iteration
		 *            current iteration (number of inference/update steps).
		 * @return
		 */
		boolean afterEpoch(Model curModel, int epoch, double loss, int iteration);

	}

}
