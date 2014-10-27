package br.pucrio.inf.learn.structlearning.discriminative.driver;

import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.OptionBuilder;
import org.apache.commons.cli.Options;
import org.apache.commons.cli.ParseException;
import org.apache.commons.cli.PosixParser;
import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;

import br.pucrio.inf.learn.structlearning.discriminative.algorithm.OnlineStructuredAlgorithm.LearnRateUpdateStrategy;
import br.pucrio.inf.learn.structlearning.discriminative.algorithm.perceptron.LossAugmentedPerceptron;
import br.pucrio.inf.learn.structlearning.discriminative.application.pq.PQInference2PBM;
import br.pucrio.inf.learn.structlearning.discriminative.application.pq.PQModel2;
import br.pucrio.inf.learn.structlearning.discriminative.application.pq.data.PQDataset2;
import br.pucrio.inf.learn.structlearning.discriminative.application.pq.data.PQOutput2;
import br.pucrio.inf.learn.structlearning.discriminative.data.ExampleInputArray;
import br.pucrio.inf.learn.structlearning.discriminative.data.encoding.FeatureEncoding;
import br.pucrio.inf.learn.structlearning.discriminative.data.encoding.StringMapEncoding;
import br.pucrio.inf.learn.structlearning.discriminative.driver.Driver.Command;
import br.pucrio.inf.learn.structlearning.discriminative.evaluation.F1Measure;
import br.pucrio.inf.learn.util.CommandLineOptionsUtil;

public class TestPQ2 implements Command {

	/**
	 * Logging object.
	 */
	private static final Log LOG = LogFactory.getLog(TestPQ2.class);

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
		String testCorpusFileName = cmdLine.getOptionValue("testcorpus");

		// Get the options given in the command-line or the corresponding
		// default values.
		String[] inputCorpusFileNames = cmdLine.getOptionValues("incorpus");

		LOG.info("Loading input corpus...");
		PQDataset2 inputCorpusA = null;

		FeatureEncoding<String> featureEncoding = null;

		try {
			/*
			 * No encoding given by the user. Create an empty and flexible
			 * feature encoding that will encode unambiguously all feature
			 * values. If the training dataset is big, this may not fit in
			 * memory.
			 */
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

		// Structure.
		LOG.info("Allocating initial model...");
		PQInference2PBM inference = new PQInference2PBM();

		// Create a new model at each iteration.
		PQModel2 model = new PQModel2(inputCorpusA.getNumberOfSymbols());

		// Training.
		LossAugmentedPerceptron alg = new LossAugmentedPerceptron(inference,
				model, numEpochs, learningRate, lossWeight, true,
				averageWeights, LearnRateUpdateStrategy.NONE);

		alg.train(inputCorpusA.getInputs(), inputCorpusA.getOutputs());

		// Test.
		// Ignore features not seen in the training corpus.
		inputCorpusA.getFeatureEncoding().setReadOnly(true);

		F1Measure eval = new F1Measure("Quotation-Person");

		try {
			LOG.info("Loading and preparing test data...");
			PQDataset2 testset = new PQDataset2(featureEncoding);
			testset.load(testCorpusFileName);

			// Allocate output sequences for predictions.
			ExampleInputArray inputs = testset.getInputs();
			PQOutput2[] outputs = testset.getOutputs();
			PQOutput2[] predicteds = new PQOutput2[outputs.length];
			for (int idx = 0; idx < outputs.length; ++idx)
				predicteds[idx] = (PQOutput2) outputs[idx].createNewObject();
			
			inputs.loadInOrder();
			
			// Fill the list of predicted outputs.
			for (int idx = 0; idx < inputs.getNumberExamples(); ++idx) {
				// Predict (tag the output sequence).
				inference.inference(model, inputs.get(idx), predicteds[idx]);

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

					/*
					 * //TO REMOVE if ((outputs[idx].getAuthor(j) != 0) &&
					 * (outputs[idx].getAuthor(j) != predicteds[idx]
					 * .getAuthor(j))) { String docId = inputs[idx].getDocId();
					 * Quotation[] quotationArray =
					 * inputs[idx].getQuotationIndexes(); int[]
					 * mispredictedQuotationIndexes =
					 * quotationArray[j].getQuotationIndex();
					 * 
					 * System.out.print("Recall "); System.out.print(docId +
					 * " "); System.out.print(mispredictedQuotationIndexes[0] +
					 * " "); System.out.print(mispredictedQuotationIndexes[1]);
					 * System.out.println(); } else if
					 * ((predicteds[idx].getAuthor(j) != 0) &&
					 * (outputs[idx].getAuthor(j) != predicteds[idx]
					 * .getAuthor(j))) { String docId = inputs[idx].getDocId();
					 * Quotation[] quotationArray =
					 * inputs[idx].getQuotationIndexes(); int[]
					 * mispredictedQuotationIndexes =
					 * quotationArray[j].getQuotationIndex();
					 * 
					 * System.out.print("Precision "); System.out.print(docId +
					 * " "); System.out.print(mispredictedQuotationIndexes[0] +
					 * " "); System.out.print(mispredictedQuotationIndexes[1]);
					 * System.out.println(); }
					 */
				}
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
}
