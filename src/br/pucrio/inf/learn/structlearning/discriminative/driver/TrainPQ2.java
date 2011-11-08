package br.pucrio.inf.learn.structlearning.discriminative.driver;

import java.util.Random;

import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.OptionBuilder;
import org.apache.commons.cli.Options;
import org.apache.commons.cli.ParseException;
import org.apache.commons.cli.PosixParser;
import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;

import br.pucrio.inf.learn.structlearning.discriminative.evaluation.F1Measure;
import br.pucrio.inf.learn.structlearning.discriminative.algorithm.OnlineStructuredAlgorithm.LearnRateUpdateStrategy;
import br.pucrio.inf.learn.structlearning.discriminative.algorithm.perceptron.Perceptron;
import br.pucrio.inf.learn.structlearning.discriminative.application.pq.data.PQDataset2;
import br.pucrio.inf.learn.structlearning.discriminative.application.pq.data.PQInput2;
import br.pucrio.inf.learn.structlearning.discriminative.application.pq.data.PQOutput2;
import br.pucrio.inf.learn.structlearning.discriminative.application.pq.PQModel2;
import br.pucrio.inf.learn.structlearning.discriminative.application.pq.PQInference2;
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
		}
		catch (Exception e) {
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
		Inference inference = new PQInference2();
		
		int numFolds = 5;
		int numExamples = inputCorpusA.getInputs().length;
		int seed = 12032041;
		Random generator = new Random(seed);
		int[] mask = new int[numExamples];
		for(int i = 0; i < numExamples; ++i) {
			mask[i] = generator.nextInt(numFolds);
		}
		
		F1Measure evalAverage = new F1Measure("Quotation-Person");
		
		for(int fold = 0; fold < numFolds; ++fold) {
			// Count the number of training examples.
			int numTrainExamples = 0;
			for(int j = 0; j < numExamples; ++j)
				if(mask[j] != fold)
					++numTrainExamples;
			
			PQInput2[] trainCorpusInput   = new PQInput2[numTrainExamples];
			PQOutput2[] trainCorpusOutput = new PQOutput2[numTrainExamples];
			PQInput2[] devCorpusInput     = new PQInput2[numExamples - numTrainExamples];
			PQOutput2[] devCorpusOutput   = new PQOutput2[numExamples - numTrainExamples];
			
			int trainIdx = 0;
			int devIdx   = 0;
			for(int j = 0; j < numExamples; ++j) {
				if(mask[j] == fold) {
					devCorpusInput[devIdx]  = inputCorpusA.getInput(j);
					devCorpusOutput[devIdx] = inputCorpusA.getOutput(j);
					++devIdx;
				}
				else {
					trainCorpusInput[trainIdx]  = inputCorpusA.getInput(j);
					trainCorpusOutput[trainIdx] = inputCorpusA.getOutput(j);
					++trainIdx;
				}
			}
			
			// Create a new model at each iteration.
			Model model = new PQModel2(inputCorpusA.getNumberOfSymbols());
			
			// Training.
			Perceptron alg = new Perceptron(inference, model, numEpochs, learningRate,
					 false, averageWeights, learningRateUpdateStrategy);
			
			alg.train(trainCorpusInput, trainCorpusOutput,
					inputCorpusA.getFeatureEncoding(), null);
			
			// Evaluate on development set.
			// Ignore features not seen in the training corpus.
			inputCorpusA.getFeatureEncoding().setReadOnly(true);

			// Allocate output sequences for predictions.
			PQInput2[] inputs = devCorpusInput;
			PQOutput2[] outputs = devCorpusOutput;
			PQOutput2[] predicteds = new PQOutput2[inputs.length];
			for (int idx = 0; idx < inputs.length; ++idx)
				predicteds[idx] = (PQOutput2) inputs[idx]
						.createOutput();
			
			F1Measure eval = new F1Measure("Quotation-Person");
			
			// Fill the list of predicted outputs.
			for (int idx = 0; idx < inputs.length; ++idx) {
				// Predict (tag the output example).
				inference.inference(model, inputs[idx], predicteds[idx]);
				// Increment data for evaluation.
				int outputsSize = outputs[idx].size();
				for(int j = 0; j < outputsSize; ++j) {
					// Total.
					if(outputs[idx].getAuthor(j) != -1) {
						eval.incNumObjects();
						evalAverage.incNumObjects();
					}
					
					// Retrieved.
					if(predicteds[idx].getAuthor(j) != -1) {
						eval.incNumPredicted();
						evalAverage.incNumPredicted();
					}
					
					// Correct.
					if((outputs[idx].getAuthor(j) != -1) &&
							(outputs[idx].getAuthor(j) == predicteds[idx].getAuthor(j))) {
						eval.incNumCorrectlyPredicted();
						evalAverage.incNumCorrectlyPredicted();
					}
				}
			}

			// Write results (precision, recall and F-1) per class.
			printF1Results("Performance at fold " + fold + ": ", eval);
		}
		
		printF1Results("Average Performance: ", evalAverage);
	}

	/**
	 * Print the given result set and title.
	 * 
	 * @param title
	 * @param results
	 */
	private static void printF1Results(String title,
			F1Measure f1) {

		// Title and header.
		System.out.println("\n" + title + "\n");
		System.out.println("|+");
		System.out
				.println("! Class !! P !! R !! F !! Total (TP+FN) !! Retrieved (TP+FP) !! Correct (TP)");

		// Overall result.
		if (f1 != null) {
			System.out.println("|-");
			System.out.println(String.format(
					"| %s || %.2f || %.2f || %.2f || %d || %d || %d",
					"overall", 100 * f1.getPrecision(), 100 * f1.getRecall(),
					100 * f1.getF1(), f1.getNumObjects(), f1.getNumRetrieved(),
					f1.getNumCorrectlyRetrieved()));
		}

		// Footer.
		System.out.println();

	}
}
