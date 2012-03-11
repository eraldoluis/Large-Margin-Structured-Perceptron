package br.pucrio.inf.learn.structlearning.discriminative.driver;

import java.util.Collection;
import java.util.Map;

import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.OptionBuilder;
import org.apache.commons.cli.Options;
import org.apache.commons.cli.ParseException;
import org.apache.commons.cli.PosixParser;
import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;

import br.pucrio.inf.learn.structlearning.discriminative.algorithm.OnlineStructuredAlgorithm.LearnRateUpdateStrategy;
import br.pucrio.inf.learn.structlearning.discriminative.algorithm.TrainingListener;
import br.pucrio.inf.learn.structlearning.discriminative.algorithm.perceptron.LossAugmentedPerceptron;
import br.pucrio.inf.learn.structlearning.discriminative.algorithm.perceptron.Perceptron;
import br.pucrio.inf.learn.structlearning.discriminative.application.coreference.CorefColumnDataset;
import br.pucrio.inf.learn.structlearning.discriminative.application.dp.DPModel;
import br.pucrio.inf.learn.structlearning.discriminative.application.dp.DPTemplateEvolutionModel;
import br.pucrio.inf.learn.structlearning.discriminative.application.dp.MaximumBranchingInference;
import br.pucrio.inf.learn.structlearning.discriminative.application.dp.data.DPDataset;
import br.pucrio.inf.learn.structlearning.discriminative.application.dp.data.DPInput;
import br.pucrio.inf.learn.structlearning.discriminative.application.dp.data.DPOutput;
import br.pucrio.inf.learn.structlearning.discriminative.application.dp.evaluation.DPEvaluation;
import br.pucrio.inf.learn.structlearning.discriminative.data.encoding.FeatureEncoding;
import br.pucrio.inf.learn.structlearning.discriminative.data.encoding.StringMapEncoding;
import br.pucrio.inf.learn.structlearning.discriminative.driver.Driver.Command;
import br.pucrio.inf.learn.structlearning.discriminative.evaluation.AccuracyEvaluation;
import br.pucrio.inf.learn.structlearning.discriminative.task.Inference;
import br.pucrio.inf.learn.structlearning.discriminative.task.Model;
import br.pucrio.inf.learn.util.CommandLineOptionsUtil;

/**
 * Driver to discriminatively train a coreference resolution model using
 * perceptron-based algorithms.
 * 
 * @author eraldo
 * 
 */
public class TrainCoreference implements Command {

	/**
	 * Logging object.
	 */
	private static final Log LOG = LogFactory.getLog(TrainCoreference.class);

	@SuppressWarnings("static-access")
	@Override
	public void run(String[] args) {
		Options options = new Options();
		options.addOption(OptionBuilder.withLongOpt("train").isRequired()
				.withArgName("filename").hasArg()
				.withDescription("Training dataset file name.").create());
		options.addOption(OptionBuilder.withLongOpt("templates").isRequired()
				.withArgName("filename").hasArg()
				.withDescription("Feature templates file name.").create());
		options.addOption(OptionBuilder
				.withLongOpt("numepochs")
				.withArgName("integer")
				.hasArg()
				.withDescription(
						"Number of epochs: how many iterations over the"
								+ " training set.").create());
		options.addOption(OptionBuilder.withLongOpt("test")
				.withArgName("filename").hasArg()
				.withDescription("Test dataset file name.").create());
		options.addOption(OptionBuilder
				.withLongOpt("testout")
				.withArgName("filename")
				.hasArg()
				.withDescription(
						"File name to save the output of the test "
								+ "dataset (with predicted feature).").create());
		options.addOption(OptionBuilder
				.withLongOpt("perepoch")
				.withDescription(
						"The evaluation on the test corpus will "
								+ "be performed after each training epoch.")
				.create());
		options.addOption(OptionBuilder
				.withLongOpt("progress")
				.withArgName("rate of examples")
				.hasArg()
				.withDescription(
						"Rate to report the training progress within each"
								+ " epoch.").create());
		options.addOption(OptionBuilder.withLongOpt("seed")
				.withArgName("integer").hasArg()
				.withDescription("Random number generator seed.").create());
		options.addOption(OptionBuilder
				.withLongOpt("lossweight")
				.withArgName("double")
				.hasArg()
				.withDescription(
						"Weight of the loss term in the inference objective"
								+ " function.").create());
		options.addOption(OptionBuilder
				.withLongOpt("noavg")
				.withDescription(
						"Turn off the weight vector averaging, i.e.,"
								+ " the algorithm returns only the final weight "
								+ "vector instead of the average of each step "
								+ "vectors.").create());

		// Parse the command-line arguments.
		CommandLine cmdLine = null;
		PosixParser parser = new PosixParser();
		try {
			cmdLine = parser.parse(options, args);
		} catch (ParseException e) {
			System.err.println(e.getMessage());
			CommandLineOptionsUtil.usage(getClass().getSimpleName(), options);
		}

		// Print the list of options along the values provided by the user.
		CommandLineOptionsUtil.printOptionValues(cmdLine, options);

		/*
		 * Get the options given in the command-line or the corresponding
		 * default values.
		 */
		String[] inputCorpusFileNames = cmdLine.getOptionValues("train");
		String templatesFileName = cmdLine.getOptionValue("templates");
		int numEpochs = Integer.parseInt(cmdLine.getOptionValue("numepochs",
				"10"));
		String testDatasetFileName = cmdLine.getOptionValue("test");
		String outputTestFileName = cmdLine.getOptionValue("testout");
		boolean evalPerEpoch = cmdLine.hasOption("perepoch");
		Double reportProgressRate = Double.parseDouble(cmdLine.getOptionValue(
				"progress", "0.1"));
		String seedStr = cmdLine.getOptionValue("seed");
		double lossWeight = Double.parseDouble(cmdLine.getOptionValue(
				"lossweight", "0d"));
		boolean averageWeights = !cmdLine.hasOption("noavg");

		CorefColumnDataset inDataset = null;
		FeatureEncoding<String> featureEncoding = null;
		try {

			/*
			 * Create an empty and flexible feature encoding that will encode
			 * unambiguously all feature values. If the training dataset is big,
			 * this may not fit in memory.
			 */
			featureEncoding = new StringMapEncoding();

			LOG.info("Loading edge corpus...");
			inDataset = new CorefColumnDataset(featureEncoding,
					(Collection<String>) null);
			inDataset.load(inputCorpusFileNames[0]);
			LOG.info("Loading templates and generating features...");
			inDataset.loadTemplates(templatesFileName, true);

			// Generate explicit features from templates.
			inDataset.generateFeatures();

		} catch (Exception e) {
			LOG.error("Parsing command-line options", e);
			System.exit(1);
		}

		// Template-based model.
		LOG.info("Allocating initial model...");
		DPModel model = new DPTemplateEvolutionModel();

		// Inference algorithm.
		MaximumBranchingInference inference = new MaximumBranchingInference(
				inDataset.getMaxNumberOfTokens());

		// Create the chosen algorithm.
		Perceptron alg = new LossAugmentedPerceptron(inference, model,
				numEpochs, 1d, lossWeight, true, averageWeights,
				LearnRateUpdateStrategy.NONE);

		if (seedStr != null)
			// User provided seed to random number generator.
			alg.setSeed(Long.parseLong(seedStr));

		if (reportProgressRate != null)
			// Progress report rate.
			alg.setReportProgressRate(reportProgressRate);

		// Ignore features not seen in the training corpus.
		featureEncoding.setReadOnly(true);

		// Evaluation method.
		DPEvaluation eval = new DPEvaluation(false);

		CorefColumnDataset testset = null;

		// Evaluation after each training epoch.
		if (testDatasetFileName != null && evalPerEpoch) {
			try {
				LOG.info("Loading and preparing test data...");
				testset = new CorefColumnDataset(inDataset);
				testset.load(testDatasetFileName);
				LOG.info("Generating features from templates...");
				testset.generateFeatures();

				alg.setListener(new EvaluateModelListener(eval, testset,
						averageWeights));
			} catch (Exception e) {
				LOG.error("Loading testset " + testDatasetFileName, e);
				System.exit(1);
			}
		} else {
			alg.setListener(new EvaluateModelListener(eval, null, false));
		}

		LOG.info("Training model...");
		// Train model.
		alg.train(inDataset.getInputs(), inDataset.getOutputs());

		// Evaluation only for the final model.
		if (testDatasetFileName != null
				&& (!evalPerEpoch || outputTestFileName != null)) {
			try {
				if (testset == null) {
					LOG.info("Loading and preparing test data...");
					testset = new CorefColumnDataset(inDataset);
					testset.load(testDatasetFileName);
					LOG.info("Generating features from templates...");
					testset.generateFeatures();
				}

				// Allocate output sequences for predictions.
				DPInput[] inputs = testset.getInputs();
				DPOutput[] outputs = testset.getOutputs();
				DPOutput[] predicteds = new DPOutput[inputs.length];
				for (int idx = 0; idx < inputs.length; ++idx)
					predicteds[idx] = (DPOutput) inputs[idx].createOutput();

				// Fill the list of predicted outputs.
				for (int idx = 0; idx < inputs.length; ++idx)
					// Predict (tag the output sequence).
					inference.inference(model, inputs[idx], predicteds[idx]);

				if (outputTestFileName != null) {
					LOG.info("Saving test file with predicted column.");
					testset.save(outputTestFileName, predicteds);
				}

				if (!evalPerEpoch) {
					// Evaluate the sequences.
					Map<String, Double> results = eval.evaluateExamples(inputs,
							outputs, predicteds);
					// Write results (precision, recall and F-1) per class.
					printAccuracyResults("Final performance:", results);
				}
			} catch (Exception e) {
				LOG.error("Loading testset " + testDatasetFileName, e);
				System.exit(1);
			}
		}

		LOG.info(String.format("# updated parameters: %d",
				model.getNumberOfUpdatedParameters()));

		LOG.info("Training done!");
	}

	/**
	 * Convert a string that can specify a value directly or in bits. If the
	 * string ends with a b, the value is considered to be n=log_2(v), where v
	 * is the specified value and n is the returned value (the value of
	 * interest).
	 * 
	 * @param valStr
	 * @return
	 */
	public static int parseValueDirectOrBits(String valStr) {
		if (valStr.endsWith("b")) {
			// The size is specified in bits.
			String bitsStr = valStr.substring(0, valStr.length() - 1);
			int bits = Integer.parseInt(bitsStr);
			return ((int) Math.round(Math.pow(2, bits))) - 1;
		}

		// The size is specified directly.
		return Integer.parseInt(valStr);
	}

	/**
	 * Print the given result set and title.
	 * 
	 * @param title
	 * @param results
	 */
	private static void printAccuracyResults(String title,
			Map<String, Double> results) {

		// Title and header.
		System.out.println("\n" + title + "\n");
		System.out.println("|+");
		System.out.println("! Type !! Accuracy");

		// Per-class results.
		System.out.println("|-");
		System.out.println(String.format("| %s || %.4f ", "Average accuracy",
				100 * results.get("average")));
		System.out.println("|-");
		System.out.println(String.format("| %s || %.4f ",
				"Per-example accuracy", 100 * results.get("example")));

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

		private AccuracyEvaluation eval;

		private DPInput[] inputs;

		private DPOutput[] outputs;

		private DPOutput[] predicteds;

		private boolean averageWeights;

		public EvaluateModelListener(AccuracyEvaluation eval,
				DPDataset testset, boolean averageWeights) {
			if (testset != null) {
				this.inputs = testset.getInputs();
				this.outputs = testset.getOutputs();
			}
			this.eval = eval;
			this.averageWeights = averageWeights;
			if (inputs != null) {
				this.predicteds = new DPOutput[inputs.length];
				// Allocate output sequences for predictions.
				for (int idx = 0; idx < inputs.length; ++idx)
					predicteds[idx] = (DPOutput) inputs[idx].createOutput();
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
				} catch (CloneNotSupportedException e) {
					LOG.error("Cloning current model on epoch " + epoch
							+ " and iteration " + iteration, e);
					return true;
				}
			}

			/*
			 * The averaged perceptron averages the final model only in the end
			 * of the training process, hence we need to average the temporary
			 * model here in order to have a better picture of its current
			 * (intermediary) performance.
			 */
			if (averageWeights)
				model.average(iteration);

			// Fill the list of predicted outputs.
			for (int idx = 0; idx < inputs.length; ++idx)
				// Predict (tag the output sequence).
				inferenceImpl.inference(model, inputs[idx], predicteds[idx]);

			// Evaluate the sequences.
			Map<String, Double> results = eval.evaluateExamples(inputs,
					outputs, predicteds);

			// Write results (precision, recall and F-1) per class.
			printAccuracyResults("Performance after epoch " + epoch + ":",
					results);

			return true;
		}

		@Override
		public void progressReport(Inference impl, Model curModel, int epoch,
				double loss, int iteration) {
		}
	}
}
