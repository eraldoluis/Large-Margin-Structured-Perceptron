package br.pucrio.inf.learn.structlearning.discriminative.algorithm.perceptron;

import java.util.Random;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;

import br.pucrio.inf.learn.structlearning.discriminative.algorithm.OnlineStructuredAlgorithm;
import br.pucrio.inf.learn.structlearning.discriminative.algorithm.TrainingListener;
import br.pucrio.inf.learn.structlearning.discriminative.application.dpgs.DPGSInference;
import br.pucrio.inf.learn.structlearning.discriminative.application.dpgs.DPGSInput;
import br.pucrio.inf.learn.structlearning.discriminative.application.dpgs.DPGSModel;
import br.pucrio.inf.learn.structlearning.discriminative.application.dpgs.DPGSOutput;
import br.pucrio.inf.learn.structlearning.discriminative.application.sequence.AveragedParameter;
import br.pucrio.inf.learn.structlearning.discriminative.application.sequence.data.SequenceInput;
import br.pucrio.inf.learn.structlearning.discriminative.application.sequence.data.SequenceOutput;
import br.pucrio.inf.learn.structlearning.discriminative.data.ExampleInput;
import br.pucrio.inf.learn.structlearning.discriminative.data.ExampleInputArray;
import br.pucrio.inf.learn.structlearning.discriminative.data.ExampleOutput;
import br.pucrio.inf.learn.structlearning.discriminative.task.Inference;
import br.pucrio.inf.learn.structlearning.discriminative.task.Model;
import br.pucrio.inf.learn.util.DebugUtil;

/**
 * Perceptron-trained linear classifier with structural examples. This class
 * implements the averaged version of the algorithm.
 * 
 * 
 */
public class Perceptron implements OnlineStructuredAlgorithm {

	// // On demand feature generation.
	// public FeatureTemplate[] templates;
	// public FeatureEncoding<Feature> encoding;

	/**
	 * Logging object.
	 */
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
	 * Number of passes over the training set.
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
	 * Training input structures.
	 */
	protected ExampleInputArray inputs;

	/**
	 * Training output structures.
	 */
	protected ExampleOutput[] outputs;

	/**
	 * Cache of predicted output structures used by the inference algorithms.
	 */
	protected ExampleOutput[] predicteds;

	/**
	 * Index of the training example in the current iteration, within the
	 * examples array given for the training method.
	 */
	protected int indexCurrentExample;

	/**
	 * An object to observe the training process.
	 */
	protected TrainingListener listener;

	/**
	 * If this value is <code>false</code>, do not randomize the order to
	 * process the training examples. This value is <code>true</code> by
	 * default.
	 */
	protected boolean randomize;

	/**
	 * Random-number generator.
	 */
	protected Random random;

	/**
	 * Report progress with this rate;
	 */
	protected double reportProgressRate;

	/**
	 * If <code>true</code> then consider partially-annotated examples.
	 */
	protected boolean partiallyAnnotatedExamples;

	/**
	 * If <code>true</code> then use the averaged implementation in which the
	 * final weight vector is equal to the average of the vector of all steps.
	 * Usually this version works better than the original algorithm that
	 * returns only the final vector.
	 */
	protected boolean averageWeights;

	/**
	 * Strategy used to vary the learning rate during training.
	 */
	protected LearnRateUpdateStrategy learningRateUpdateStrategy;

	/**
	 * Indexes that give the examples training order.
	 */
	protected int[] indexTrainingOrder;

	/**
	 * Create a perceptron to train the given initial model using the default
	 * Collins' learning rate (1) and the default number of iterations (10).
	 * 
	 * @param inferenceImpl
	 * @param initialModel
	 */
	public Perceptron(Inference inferenceImpl, Model initialModel) {
		this(inferenceImpl, initialModel, 10, 1d, true, true,
				LearnRateUpdateStrategy.NONE);
	}

	/**
	 * Create a perceptron to train the given initial model using the given
	 * number of iterations and learning rate.
	 * 
	 * @param inferenceImpl
	 * @param initialModel
	 * @param numberOfEpochs
	 * @param learningRate
	 * @param randomize
	 * @param averageWeights
	 * @param learningRateUpdateStrategy
	 */
	public Perceptron(Inference inferenceImpl, Model initialModel,
			int numberOfEpochs, double learningRate, boolean randomize,
			boolean averageWeights,
			LearnRateUpdateStrategy learningRateUpdateStrategy) {
		this.inferenceImpl = inferenceImpl;
		this.model = initialModel;
		this.numberOfEpochs = numberOfEpochs;
		this.learningRate = learningRate;
		this.random = new Random();
		this.randomize = randomize;
		this.averageWeights = averageWeights;
		this.learningRateUpdateStrategy = learningRateUpdateStrategy;
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

	@Override
	public Model getModel() {
		return model;
	}

	public boolean getAverageWeights() {
		return averageWeights;
	}

	@Override
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
	public void setListener(TrainingListener listener) {
		this.listener = listener;
	}

	/**
	 * Return the learning rate for the current iteration according to the
	 * chosen update strategy.
	 * 
	 * @return
	 */
	protected double getCurrentLearningRate() {
		switch (learningRateUpdateStrategy) {
		case NONE:
			return learningRate;
		case LINEAR:
			return learningRate / (iteration + 1);
		case QUADRATIC:
			return learningRate / ((iteration + 1) * (learningRate + 1));
		case SQUARE_ROOT:
			return learningRate / Math.sqrt(iteration + 1);
		default:
			return learningRate;
		}
	}

	public void train(ExampleInputArray inputs, ExampleOutput[] outputs) {
		// Training examples.
		this.inputs = inputs;
		this.outputs = outputs;

		int numbersExamples = inputs.getNumberExamples();

		// Allocate predicted output objects for the training example.
		predicteds = new ExampleOutput[outputs.length];
		for (int idx = 0; idx < numbersExamples; ++idx)
			predicteds[idx] = outputs[idx].createNewObject();

		// Examples training order.
		indexTrainingOrder = new int[numbersExamples];
		for (int idx = 0; idx < numbersExamples; ++idx)
			indexTrainingOrder[idx] = idx;

		if (listener != null)
			if (!listener.beforeTraining(inferenceImpl, model))
				return;

		iteration = 0;
		for (epoch = 0; epoch < numberOfEpochs; ++epoch) {

			LOG.info("Perceptron epoch: " + epoch + "...");

			if (listener != null)
				if (!listener.beforeEpoch(inferenceImpl, model, epoch,
						iteration))
					// Stop training.
					break;

			// Train one epoch and get the accumulated loss.
			double loss = trainOneEpoch(inputs, outputs, predicteds);

			LOG.info("Training loss: " + loss);
			LOG.info("Normalized training loss: " + (loss / numbersExamples));

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
		if (averageWeights)
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
	 * @return the sum of the losses over all examples through this epoch
	 */
	public double trainOneEpoch(ExampleInputArray inputs,
			ExampleOutput[] outputs, ExampleOutput[] predicteds) {

		// Accumulate the loss over all examples in this epoch.
		double loss = 0d;

		// Progress report.
		int reportProgressInterval = (int) (inputs.getNumberExamples() * reportProgressRate);
		if (reportProgressInterval > 0)
			System.out.print("Progress: ");

		if (randomize) {
			// Randomize the order to process the training examples.
			int len = indexTrainingOrder.length;
			for (int idx = len - 1; idx >= 0; --idx) {
				int idxSwp = random.nextInt(len - idx);
				int tmp = indexTrainingOrder[idxSwp];
				indexTrainingOrder[idxSwp] = indexTrainingOrder[idx];
				indexTrainingOrder[idx] = tmp;
			}
		}

		inputs.load(indexTrainingOrder);

		// Iterate over the training examples, updating the weight vector.
		for (int idx = 0; idx < inputs.getNumberExamples(); ++idx) {

			indexCurrentExample = indexTrainingOrder[idx];
			/*
			 * Update the current model weights according with the predicted
			 * output for this training example.
			 */

			ExampleInput input = inputs.get(indexCurrentExample);

			double loss1 = train(input, outputs[indexCurrentExample],
					predicteds[indexCurrentExample]);

			loss += loss1;

			// Progress report.
			if (reportProgressInterval > 0
					&& (idx + 1) % reportProgressInterval == 0) {
				LOG.info(Math.round((idx + 1) * 100d
						/ inputs.getNumberExamples())
						+ "% ");
				// Progress report listener.
				if (listener != null)
					listener.progressReport(inferenceImpl, model, epoch, loss,
							iteration);
			}
		}

		// Progress report.
		if (reportProgressInterval > 0)
			System.out.println("done.");

		return loss;

	}

	@Override
	public void train(ExampleInputArray inputsA, ExampleOutput[] outputsA,
			double weightA, double weightStep, ExampleInputArray inputsB,
			ExampleOutput[] outputsB) {

		// Allocate predicted output objects for the training examples.
		predicteds = new ExampleOutput[outputsA.length];
		for (int idx = 0; idx < outputsA.length; ++idx)
			predicteds[idx] = outputsA[idx].createNewObject();
		ExampleOutput[] predictedsB = new ExampleOutput[outputsB.length];
		for (int idx = 0; idx < outputsB.length; ++idx)
			predictedsB[idx] = outputsB[idx].createNewObject();

		if (listener != null)
			if (!listener.beforeTraining(inferenceImpl, model))
				return;

		iteration = 0;
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
			double loss = trainOneEpoch(inputsA, outputsA, predicteds,
					epochWeightA, inputsB, outputsB, predictedsB);

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
		if (averageWeights)
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
	 * @return
	 */
	public double trainOneEpoch(ExampleInputArray inputsA,
			ExampleOutput[] outputsA, ExampleOutput[] predictedsA,
			double weightA, ExampleInputArray inputsB,
			ExampleOutput[] outputsB, ExampleOutput[] predictedsB) {

		LOG.info("Weight of first dataset in this epoch: " + weightA);

		// Accumulate the loss over all examples in this epoch.
		double loss = 0d;

		/*
		 * Use only the dataset A length to determine the length of an epoch.
		 * This allows some easier comparison settings (per-epoch plots, for
		 * instance) among models trained with different B datasets but only one
		 * A dataset.
		 */
		int totalLength = inputsA.getNumberExamples();

		// Progress report.
		int reportProgressInterval = (int) (totalLength * reportProgressRate);
		if (reportProgressInterval > 0)
			System.out.print("Progress: ");

		// Iterate over the training examples, updating the weight vector.
		for (int idx = 0, idxA = 0, idxB = 0; idx < totalLength; ++idx) {

			// Randomly choose from A or B datasets.
			double aOrB = random.nextDouble();

			if (aOrB <= weightA) {

				// Train on A example.
				int idxEx = idxA++;

				if (randomize)
					// Randomize the order to process the training examples.
					idxEx = random.nextInt(inputsA.getNumberExamples());

				inputsA.load(new int[idxEx]);
				/*
				 * Update the current model weights according with the predicted
				 * output for this training example.
				 */
				loss += train(inputsA.get(idxEx), outputsA[idxEx],
						predictedsA[idxEx]);

			} else {

				// Train on B example.
				int idxEx = idxB++;

				if (randomize)
					// Randomize the order to process the training examples.
					idxEx = random.nextInt(inputsB.getNumberExamples());

				inputsB.load(new int[idxEx]);

				/*
				 * Update the current model weights according with the predicted
				 * output for this training example.
				 */
				loss += train(inputsB.get(idxEx), outputsB[idxEx],
						predictedsB[idxEx]);

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

	@Override
	public double train(ExampleInput input, ExampleOutput correctOutput,
			ExampleOutput predictedOutput) {

		ExampleOutput referenceOutput = correctOutput;
		if (partiallyAnnotatedExamples) {
			/*
			 * If the user asked to consider partially-labeled examples then
			 * infer the missing values within the given correct output
			 * structure before updating the current model.
			 */
			referenceOutput = correctOutput.createNewObject();
			inferenceImpl.partialInference(model, input, correctOutput,
					referenceOutput);
		}

		// Predict the best output with the current mobel.
		inferenceImpl.inference(model, input, predictedOutput);

		// Update the current model and return the loss for this example.
		double loss = model.update(input, referenceOutput, predictedOutput,
				getCurrentLearningRate());

		// Debug.
		if (DebugUtil.print)
			DebugUtil.printSequence((SequenceInput) input,
					(SequenceOutput) referenceOutput,
					(SequenceOutput) predictedOutput, loss);

		// Averaged-Perceptron: account the updates into the averaged weights.
		model.sumUpdates(iteration);

		++iteration;

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

	@Override
	public int getIteration() {
		return iteration;
	}
}
