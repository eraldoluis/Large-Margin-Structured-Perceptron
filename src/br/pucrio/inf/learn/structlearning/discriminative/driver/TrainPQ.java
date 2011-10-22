package br.pucrio.inf.learn.structlearning.discriminative.driver;

import java.util.Map;
import java.util.Map.Entry;

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
import br.pucrio.inf.learn.structlearning.discriminative.application.pq.data.PQDataset;
import br.pucrio.inf.learn.structlearning.discriminative.application.pq.data.PQInput;
import br.pucrio.inf.learn.structlearning.discriminative.application.pq.data.PQOutput;
import br.pucrio.inf.learn.structlearning.discriminative.application.pq.PQModel;
import br.pucrio.inf.learn.structlearning.discriminative.application.pq.PQInference;
import br.pucrio.inf.learn.structlearning.discriminative.data.FeatureEncoding;
import br.pucrio.inf.learn.structlearning.discriminative.data.StringMapEncoding;
import br.pucrio.inf.learn.structlearning.discriminative.driver.Driver.Command;
import br.pucrio.inf.learn.structlearning.discriminative.task.Inference;
import br.pucrio.inf.learn.structlearning.discriminative.task.Model;
import br.pucrio.inf.learn.util.CommandLineOptionsUtil;

public class TrainPQ implements Command {

	/**
	 * Logging object.
	 */
	private static final Log LOG = LogFactory.getLog(TrainPQ.class);

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

		/*
		 * Get the options given in the command-line or the corresponding
		 * default values.
		 */
		String[] inputCorpusFileNames = cmdLine.getOptionValues("incorpus");
		
		LOG.info("Loading input corpus...");
		PQDataset inputCorpusA = null;
		
		FeatureEncoding<String> featureEncoding = null;
		
		try {
			/*
			 * No encoding given by the user. Create an empty and
			 * flexible feature encoding that will encode unambiguously
			 * all feature values. If the training dataset is big, this
			 * may not fit in memory.
			 */
			featureEncoding = new StringMapEncoding();
			
			LOG.info("Feature encoding: "
					+ featureEncoding.getClass().getSimpleName());
			
			// Get the list of input paths and concatenate the corpora in them.
			inputCorpusA = new PQDataset(featureEncoding, true);
			
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
		LOG.info("Allocating initial model...");
		Inference inference = new PQInference();

		Model model = new PQModel(inputCorpusA.getNumberOfSymbols());

		F1Measure eval = new F1Measure("Quotation-Person");

		Perceptron alg = new Perceptron(inference, model, numEpochs, learningRate,
							 false, averageWeights, learningRateUpdateStrategy);


		// Training.
		alg.train(inputCorpusA.getInputs(), inputCorpusA.getOutputs(),
				inputCorpusA.getFeatureEncoding(), null);


		// Test.
		// Ignore features not seen in the training corpus.
		inputCorpusA.getFeatureEncoding().setReadOnly(true);

		try {
			LOG.info("Loading and preparing test data...");
			PQDataset testset = new PQDataset(featureEncoding);
			testset.load(testCorpusFileName);

			// Allocate output sequences for predictions.
			PQInput[] inputs = testset.getInputs();
			PQOutput[] outputs = testset.getOutputs();
			PQOutput[] predicteds = new PQOutput[inputs.length];
			for (int idx = 0; idx < inputs.length; ++idx)
				predicteds[idx] = (PQOutput) inputs[idx]
						.createOutput();

			// Fill the list of predicted outputs.
			for (int idx = 0; idx < inputs.length; ++idx) {
				// Predict (tag the output sequence).
				inference.inference(model, inputs[idx], predicteds[idx]);
				// Increment data for evaluation.
				eval.incNumObjects();
				eval.incNumPredicted();
				if (predicteds[idx].getPerson() == outputs[idx].getPerson())
					eval.incNumCorrectlyPredicted();
			}

			// Write results (precision, recall and F-1) per class.
			printF1Results("Final performance:", eval);

		} catch (Exception e) {
			LOG.error("Loading testset " + testCorpusFileName, e);
			System.exit(1);
		}
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
