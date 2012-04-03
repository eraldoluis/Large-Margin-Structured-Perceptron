package br.pucrio.inf.learn.structlearning.discriminative.driver;

import java.util.ArrayList;
import java.util.Random;

import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.OptionBuilder;
import org.apache.commons.cli.Options;
import org.apache.commons.cli.ParseException;
import org.apache.commons.cli.PosixParser;
import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;

import br.pucrio.inf.learn.structlearning.discriminative.evaluation.F1Measure;
import br.pucrio.inf.learn.structlearning.discriminative.algorithm.TrainingListener;
import br.pucrio.inf.learn.structlearning.discriminative.algorithm.OnlineStructuredAlgorithm.LearnRateUpdateStrategy;
import br.pucrio.inf.learn.structlearning.discriminative.algorithm.perceptron.LossAugmentedPerceptron;
import br.pucrio.inf.learn.structlearning.discriminative.application.pq.data.PQDataset2;
import br.pucrio.inf.learn.structlearning.discriminative.application.pq.data.PQInput2;
import br.pucrio.inf.learn.structlearning.discriminative.application.pq.data.PQOutput2;
import br.pucrio.inf.learn.structlearning.discriminative.application.pq.PQModel2;
import br.pucrio.inf.learn.structlearning.discriminative.application.pq.PQInference2PBM;
import br.pucrio.inf.learn.structlearning.discriminative.data.FeatureEncoding;
import br.pucrio.inf.learn.structlearning.discriminative.data.StringMapEncoding;
import br.pucrio.inf.learn.structlearning.discriminative.driver.Driver.Command;
import br.pucrio.inf.learn.structlearning.discriminative.task.Inference;
import br.pucrio.inf.learn.structlearning.discriminative.task.Model;
import br.pucrio.inf.learn.util.CommandLineOptionsUtil;

public class TrainPQ2 implements Command {

	/**
	 * Logging object.
	 */
	private static final Log LOG = LogFactory.getLog(TrainPQ2.class);

	@SuppressWarnings("static-access")
	@Override
	public void run(String[] args) {
		Options options = new Options();
		options.addOption(OptionBuilder.withLongOpt("incorpus").isRequired()
				.withArgName("input corpus").hasArg()
				.withDescription("Input corpus file name.").create('i'));
		options.addOption(OptionBuilder.withLongOpt("testcorpus")
				.withArgName("test corpus").hasArg()
				.withDescription("Test corpus file name.").create('t'));
		options.addOption(OptionBuilder.withLongOpt("learnrate")
				.withArgName("learning rate within [0:1]").hasArg()
				.withDescription("Learning rate used in the updates.").create());
		options.addOption(OptionBuilder
				.withLongOpt("numepochs")
				.withArgName("number of epochs")
				.hasArg()
				.withDescription(
						"Number of epochs: how many iterations over the"
								+ " training set.").create('T'));
		options.addOption(OptionBuilder
				.withLongOpt("noavg")
				.withDescription(
						"Turn off the weight vector averaging, i.e.,"
								+ " the algorithm returns only the final weight "
								+ "vector instead of the average of each step "
								+ "vectors.").create());
		options.addOption(OptionBuilder
				.withLongOpt("lrupdate")
				.withArgName("none | linear | quadratic | root")
				.hasArg()
				.withDescription(
						"Which learning rate update strategy to be used. Valid "
								+ "values are: "
								+ "none (constant learning rate), "
								+ "linear (n/t), "
								+ "quadratic (n/(t*t)) or "
								+ "root (n/sqrt(t)), "
								+ "where n is the initial learning rate and t "
								+ "is the current iteration (number of processed"
								+ " examples).").create());
		options.addOption(OptionBuilder
				.withLongOpt("lossweight")
				.withArgName("numeric loss weight")
				.hasArg()
				.withDescription(
						"Weight of the loss term in the inference objective"
								+ " function.").create());

		// Parse the command-line arguments.
		CommandLine cmdLine = null;
		PosixParser parser = new PosixParser();
		try {
			cmdLine = parser.parse(options, args);
		} catch (ParseException e) {
			System.err.println(e.getMessage());
			CommandLineOptionsUtil.usage(getClass().getSimpleName(), options);
		}

		// List of options along the values provided by the user.
		CommandLineOptionsUtil.printOptionValues(cmdLine, options);
		double learningRate = Double.parseDouble(cmdLine.getOptionValue(
				"learnrate", "1"));
		int numEpochs = Integer.parseInt(cmdLine.getOptionValue("numepochs",
				"10"));
		boolean averageWeights = !cmdLine.hasOption("noavg");
		double lossWeight = Double.parseDouble(cmdLine.getOptionValue(
				"lossweight", "0d"));
		String lrUpdateStrategy = cmdLine.getOptionValue("lrupdate");
		String testCorpusFileName = cmdLine.getOptionValue("testcorpus");
		
		// Get the options given in the command-line or the corresponding
		// default values.
		String[] inputCorpusFileNames = cmdLine.getOptionValues("incorpus");

		LOG.info("Loading input corpus...");
		PQDataset2 inputCorpusA = null;

		FeatureEncoding<String> featureEncoding = null;

		try {
			// No encoding given by the user. Create an empty and
			// flexible feature encoding that will encode unambiguously
			// all feature values. If the training dataset is big, this
			// may not fit in memory.
			featureEncoding = new StringMapEncoding();

			LOG.info("Feature encoding: "
					+ featureEncoding.getClass().getSimpleName());

			// Get the list of input paths and concatenate the corpora in them.
			inputCorpusA = new PQDataset2(featureEncoding, true);

			// Load the first data file, which can be the standard input.
			if (inputCorpusFileNames[0].equals("stdin"))
				inputCorpusA.load(System.in);
			else
				inputCorpusA.load(inputCorpusFileNames[0]);
		} catch (Exception e) {
			LOG.error("Parsing command-line options", e);
			System.exit(1);
		}

		LOG.info("Feature encoding size: " + featureEncoding.size());

		// Learning rate update strategy.
		LearnRateUpdateStrategy learningRateUpdateStrategy = LearnRateUpdateStrategy.NONE;
		if (lrUpdateStrategy == null)
			learningRateUpdateStrategy = LearnRateUpdateStrategy.NONE;
		else if (lrUpdateStrategy.equals("none"))
			learningRateUpdateStrategy = LearnRateUpdateStrategy.NONE;
		else if (lrUpdateStrategy.equals("linear"))
			learningRateUpdateStrategy = LearnRateUpdateStrategy.LINEAR;
		else if (lrUpdateStrategy.equals("quadratic"))
			learningRateUpdateStrategy = LearnRateUpdateStrategy.QUADRATIC;
		else if (lrUpdateStrategy.equals("root"))
			learningRateUpdateStrategy = LearnRateUpdateStrategy.SQUARE_ROOT;
		else {
			System.err.println("Unknown learning rate update strategy: "
					+ lrUpdateStrategy);
			System.exit(1);
		}

		// Structure.
		PQInference2PBM inference = new PQInference2PBM();

		// Establish the number of folds and generate the fold mask.
		int numFolds = 5;
		int numExamples = inputCorpusA.getInputs().length;
		Random generator = new Random();
		int[] mask = new int[numExamples];
		for (int i = 0; i < numExamples; ++i) {
			mask[i] = generator.nextInt(numFolds);
		}

		F1Measure evalAverage = new F1Measure("Quotation-Person");
		
		
		/*
		// TO REMOVE
		ArrayList authors;
		int moreThanThreeQuotationsThatBelongToSameAuthor = 0;
		int numOfAuthorsThatHasMoreThanThreeQuotations = 0;
		*/
		
		
		// For each fold, train and evaluate the model.
		for (int fold = 0; fold < numFolds; ++fold) {
			// Count the number of training examples.
			int numTrainExamples = 0;
			for (int j = 0; j < numExamples; ++j)
				if (mask[j] != fold)
					++numTrainExamples;

			PQInput2[] trainCorpusInput   = new PQInput2[numTrainExamples];
			PQOutput2[] trainCorpusOutput = new PQOutput2[numTrainExamples];
			PQInput2[] devCorpusInput     = new PQInput2[numExamples
					- numTrainExamples];
			PQOutput2[] devCorpusOutput   = new PQOutput2[numExamples
					- numTrainExamples];

			int trainIdx = 0;
			int devIdx = 0;
			for (int j = 0; j < numExamples; ++j) {
				if (mask[j] == fold) {
					devCorpusInput[devIdx] = inputCorpusA.getInput(j);
					devCorpusOutput[devIdx] = inputCorpusA.getOutput(j);
					++devIdx;
				} else {
					trainCorpusInput[trainIdx] = inputCorpusA.getInput(j);
					trainCorpusOutput[trainIdx] = inputCorpusA.getOutput(j);
					++trainIdx;
				}
			}

			// Create a new model at each iteration.
			PQModel2 model = new PQModel2(inputCorpusA.getNumberOfSymbols());

			// Training.
			LossAugmentedPerceptron alg = new LossAugmentedPerceptron(
					inference, model, numEpochs, learningRate, lossWeight,
					true, averageWeights, learningRateUpdateStrategy);

			alg.setListener(new EvaluateModelListener(devCorpusInput, devCorpusOutput, averageWeights));

			alg.train(trainCorpusInput, trainCorpusOutput,
					inputCorpusA.getFeatureEncoding(), null);

			// Evaluate on the fold development set.
			//// Ignore features not seen in the training corpus.
			//inputCorpusA.getFeatureEncoding().setReadOnly(true);

			// Allocate output sequences for predictions.
			PQInput2[] inputs      = devCorpusInput;
			PQOutput2[] outputs    = devCorpusOutput;
			PQOutput2[] predicteds = new PQOutput2[inputs.length];
			for (int idx = 0; idx < inputs.length; ++idx)
				predicteds[idx]    = (PQOutput2) inputs[idx].createOutput();

			F1Measure eval = new F1Measure("Quotation-Person");
			// Fill the list of predicted outputs.
			for (int idx = 0; idx < inputs.length; ++idx) {
				// Predict (tag the output example).
				inference.inference(model, inputs[idx], predicteds[idx]);
				// Increment data for evaluation.
				int outputsSize = outputs[idx].size();
				
				
				/*
				//TO REMOVE
				authors = new ArrayList();
				*/
				
				
				for (int j = 0; j < outputsSize; ++j) {
					// Total.
					if (outputs[idx].getAuthor(j) != 0) {
						eval.incNumObjects();
						evalAverage.incNumObjects();
						
						
						/*
						//TO REMOVE
						if(!authors.contains(outputs[idx].getAuthor(j))) {
							authors.add(outputs[idx].getAuthor(j));
							
							int numOfQuotationsOfThisAuthor = 0;
							for(int k = 0; k < outputsSize; ++k)
								if (outputs[idx].getAuthor(k) == outputs[idx].getAuthor(j))
									++numOfQuotationsOfThisAuthor;
							if(numOfQuotationsOfThisAuthor > 3) {
								moreThanThreeQuotationsThatBelongToSameAuthor += numOfQuotationsOfThisAuthor;
								++numOfAuthorsThatHasMoreThanThreeQuotations;
							}
						}
						*/
						
						
					}

					// Retrieved.
					if (predicteds[idx].getAuthor(j) != 0) {
						eval.incNumPredicted();
						evalAverage.incNumPredicted();
					}

					// Correct.
					if ((outputs[idx].getAuthor(j) != 0)
							&& (outputs[idx].getAuthor(j) == predicteds[idx]
									.getAuthor(j))) {
						eval.incNumCorrectlyPredicted();
						evalAverage.incNumCorrectlyPredicted();
					}
				}
			}
			
			//// Write results (precision, recall and F-1) per class.
			//printF1Results("Performance at fold " + fold + ": ", eval);
			//System.out.println("-------------------------------------");
		}
		
		
		/*
		//TO REMOVE
		System.out.println("Quotations:                                                               " + evalAverage.getNumObjects());
		System.out.println("Authors that have more than 3 quotations:                                 " + numOfAuthorsThatHasMoreThanThreeQuotations);
		System.out.println("Quotations that belong to the group of more than 3 quotations per author: " + moreThanThreeQuotationsThatBelongToSameAuthor);
		System.out.println("Lost quotations:                                                          " + (moreThanThreeQuotationsThatBelongToSameAuthor - numOfAuthorsThatHasMoreThanThreeQuotations));
		*/
		
		
		//printF1Results("Average Performance: ", evalAverage);
	}

	/**
	 * Print the given result set and title.
	 * 
	 * @param title
	 * @param results
	 */
	private static void printF1Results(String title, F1Measure f1) {

		// Title and header.
		System.out.println("\n" + title + "\n");
		System.out.println("|+");
		System.out
				.println("! Class !! P !! R !! F !! Total (TP+FN) !! Retrieved (TP+FP) !! Correct (TP)");

		// Overall result.
		if (f1 != null) {
			System.out.println("|-");
			System.out.println(String.format(
					"| overall || %.2f || %.2f || %.2f || %d || %d || %d",
					100 * f1.getPrecision(), 100 * f1.getRecall(),
					100 * f1.getF1(), f1.getNumObjects(), f1.getNumRetrieved(),
					f1.getNumCorrectlyRetrieved()));
		}

		// Footer.
		System.out.println();

	}
	
	
	/**
	 * Training listener to evaluate models after each iteration.
	 * 
	 * @author eraldof
	 * 
	 */
	private static class EvaluateModelListener implements TrainingListener {

		private PQInput2[] inputs;

		private PQOutput2[] outputs;

		private PQOutput2[] predicteds;

		private boolean averageWeights;

		public EvaluateModelListener(PQInput2[] inputs, PQOutput2[] outputs,
				boolean averageWeights) {
			this.inputs = inputs;
			this.outputs = outputs;
			this.averageWeights = averageWeights;
			if (inputs != null) {
				this.predicteds = new PQOutput2[inputs.length];
				// Allocate output sequences for predictions.
				for (int idx = 0; idx < inputs.length; ++idx)
					predicteds[idx] = (PQOutput2) inputs[idx]
							.createOutput();
			}
		}

		@Override
		public boolean beforeTraining(Inference impl, Model curModel) {
			return true;
		}

		@Override
		public void afterTraining(Inference impl, Model curModel) {
		}

		@Override
		public boolean beforeEpoch(Inference impl, Model curModel, int epoch,
				int iteration) {
			return true;
		}

		@Override
		public boolean afterEpoch(Inference inferenceImpl, Model model,
				int epoch, double loss, int iteration) {

			if (inputs == null)
				return true;

			if (averageWeights) {
				try {

					// Clone the current model to average it, if necessary.
					model = (Model) model.clone();

					// The averaged perceptron averages the final model only in
					// the end of the training process, hence we need to average
					// the temporary model here in order to have a better
					// picture of its current (intermediary) performance.
					model.average(iteration);

				} catch (CloneNotSupportedException e) {
					LOG.error("Cloning current model on epoch " + epoch
							+ " and iteration " + iteration, e);
					return true;
				}
			}

			F1Measure eval = new F1Measure("Quotation-Person");

			// Fill the list of predicted outputs.
			for (int idx = 0; idx < inputs.length; ++idx) {
				// Predict (tag the output example).
				//inferenceImpl.lossAugmentedInference(model, inputs[idx], outputs[idx], predicteds[idx], loss);
				inferenceImpl.inference(model, inputs[idx], predicteds[idx]);
				// Increment data for evaluation.
				int outputsSize = outputs[idx].size();
				for (int j = 0; j < outputsSize; ++j) {
					// Total.
					if (outputs[idx].getAuthor(j) != 0)
						eval.incNumObjects();

					// Retrieved.
					if (predicteds[idx].getAuthor(j) != 0)
						eval.incNumPredicted();

					// Correct.
					if ((outputs[idx].getAuthor(j) != 0)
							&& (outputs[idx].getAuthor(j) == predicteds[idx]
									.getAuthor(j)))
						eval.incNumCorrectlyPredicted();
				}
			}

			// Write results (precision, recall and F-1) per class.
			if(epoch < 10)
				printF1Results("Performance after epoch 0" + epoch + ":", eval);
			else
				printF1Results("Performance after epoch " + epoch + ":", eval);

			return true;
		}

		@Override
		public void progressReport(Inference impl, Model curModel, int epoch,
				double loss, int iteration) {
		}
	}
}
