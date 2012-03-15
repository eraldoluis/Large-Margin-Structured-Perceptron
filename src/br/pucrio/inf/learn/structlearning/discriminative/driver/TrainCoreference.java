package br.pucrio.inf.learn.structlearning.discriminative.driver;

import java.io.BufferedReader;
import java.io.File;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.Collection;

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
import br.pucrio.inf.learn.structlearning.discriminative.application.dp.data.DPColumnDataset;
import br.pucrio.inf.learn.structlearning.discriminative.application.dp.data.DPInput;
import br.pucrio.inf.learn.structlearning.discriminative.application.dp.data.DPOutput;
import br.pucrio.inf.learn.structlearning.discriminative.data.DatasetException;
import br.pucrio.inf.learn.structlearning.discriminative.data.encoding.FeatureEncoding;
import br.pucrio.inf.learn.structlearning.discriminative.data.encoding.StringMapEncoding;
import br.pucrio.inf.learn.structlearning.discriminative.driver.Driver.Command;
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
		options.addOption(OptionBuilder.withLongOpt("scriptpath")
				.withArgName("path").hasArg()
				.withDescription("Base path for CoNLL and Python scripts.")
				.create());
		options.addOption(OptionBuilder.withLongOpt("conlltest")
				.withArgName("filename").hasArg()
				.withDescription("Test dataset on CoNLL format.").create());
		options.addOption(OptionBuilder
				.withLongOpt("conllmetric")
				.withArgName("")
				.hasArg()
				.withDescription(
						"Evaluation metric:\n"
								+ "  muc: MUCScorer (Vilain et al, 1995)\n"
								+ "  bcub: B-Cubed (Bagga and Baldwin, 1998)\n"
								+ "  ceafm: CEAF (Luo et al, 2005) using mention-based similarity\n"
								+ "  ceafe: CEAF (Luo et al, 2005) using entity-based similarity\n"
								+ "  blanc: BLANC (Recasens and Hovy, to appear)\n"
								+ "  all: uses all the metrics to score")
				.create());
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
		String scriptBasePathStr = cmdLine.getOptionValue("scriptpath");
		File conllBasePath = null;
		if (scriptBasePathStr != null)
			conllBasePath = new File(scriptBasePathStr);
		String conllTestFileName = cmdLine.getOptionValue("conlltest");
		String metric = cmdLine.getOptionValue("conllmetric");
		if (metric == null)
			metric = "all";
		boolean evalPerEpoch = cmdLine.hasOption("perepoch");
		Double reportProgressRate = Double.parseDouble(cmdLine.getOptionValue(
				"progress", "0.1"));
		String seedStr = cmdLine.getOptionValue("seed");
		double lossWeight = Double.parseDouble(cmdLine.getOptionValue(
				"lossweight", "0d"));
		boolean averageWeights = !cmdLine.hasOption("noavg");

		/*
		 * If --test is provided, then --conlltest must be provided (and
		 * vice-versa).
		 */
		if ((testDatasetFileName == null) != (conllTestFileName == null)) {
			LOG.error("if --test is provided, then --conlltest"
					+ " must be provided (and vice-versa)");
			System.exit(1);
		}

		CorefColumnDataset inDataset = null;
		FeatureEncoding<String> featureEncoding = null;
		try {
			/*
			 * Create an empty and flexible feature encoding that will encode
			 * unambiguously all feature values. If the training dataset is big,
			 * this may not fit in memory.
			 */
			featureEncoding = new StringMapEncoding();

			LOG.info("Loading train dataset...");
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
		// featureEncoding.setReadOnly(true);

		CorefColumnDataset testset = null;

		// Evaluation after each training epoch.
		if (testDatasetFileName != null && evalPerEpoch) {
			try {
				LOG.info("Loading and preparing test dataset...");
				testset = new CorefColumnDataset(inDataset);
				testset.setCheckMultipleTrueEdges(false);
				testset.load(testDatasetFileName);
				LOG.info("Generating features from templates...");
				testset.generateFeatures();
				// Predicted test set filename.
				String testPredictedFileName = testDatasetFileName + ".pred";
				// Set listener that perform evaluation after each epoch.
				alg.setListener(new EvaluateModelListener(conllBasePath,
						testPredictedFileName, conllTestFileName, metric,
						testset, averageWeights));
			} catch (Exception e) {
				LOG.error("Loading testset " + testDatasetFileName, e);
				System.exit(1);
			}
		}

		LOG.info("Training model...");
		// Train model.
		alg.train(inDataset.getInputs(), inDataset.getOutputs());

		// Evaluation only for the final model.
		if (testDatasetFileName != null && !evalPerEpoch) {
			try {

				LOG.info("Loading and preparing test dataset...");
				testset = new CorefColumnDataset(inDataset);
				testset.setCheckMultipleTrueEdges(false);
				testset.load(testDatasetFileName);
				LOG.info("Generating features from templates...");
				testset.generateFeatures();

			} catch (Exception e) {
				LOG.error("Loading testset " + testDatasetFileName, e);
				System.exit(1);
			}

			// Allocate output sequences for predictions.
			DPInput[] inputs = testset.getInputs();
			DPOutput[] predicteds = new DPOutput[inputs.length];
			for (int idx = 0; idx < inputs.length; ++idx)
				predicteds[idx] = (DPOutput) inputs[idx].createOutput();

			// Fill the list of predicted outputs.
			for (int idx = 0; idx < inputs.length; ++idx)
				// Predict (tag the output sequence).
				inference.inference(model, inputs[idx], predicteds[idx]);

			// Predicted test set filename.
			String testPredictedFileName = testDatasetFileName + ".pred";

			try {
				LOG.info("Saving test file (" + testPredictedFileName
						+ ") with predicted column...");
				testset.save(testPredictedFileName, predicteds);
			} catch (Exception e) {
				LOG.error("Saving predicted file " + testPredictedFileName, e);
				System.exit(1);
			}

			try {
				LOG.info("Final evaluation:");
				evaluateWithConllScripts(conllBasePath, testPredictedFileName,
						conllTestFileName, metric);
			} catch (Exception e) {
				LOG.error("Running evaluation scripts", e);
				System.exit(1);
			}
		}

		LOG.info(String.format("# updated parameters: %d",
				model.getNumberOfUpdatedParameters()));

		LOG.info("Training done!");
	}

	/**
	 * Execute CoNLL evaluation scripts and print results.
	 * 
	 * @param scriptBasePath
	 * @param testPredictedFileName
	 * @param conllTestFileName
	 * @param metric
	 * @throws IOException
	 * @throws CommandException
	 * @throws InterruptedException
	 */
	private static void evaluateWithConllScripts(File scriptBasePath,
			String testPredictedFileName, String conllTestFileName,
			String metric) throws IOException, CommandException,
			InterruptedException {
		// File name of CoNLL-format predicted file.
		String testConllPredictedFileName = testPredictedFileName + ".conll";
		// Command to convert the predicted file to CoNLL format.
		String cmd = String.format(
				"python mentionPairsToConll.py %s %s %s predicted",
				conllTestFileName, testPredictedFileName,
				testConllPredictedFileName);
		execCommandAndRedirectOutputAndError(cmd, new File(scriptBasePath,
				"source"));

		// Command to evaluate the predicted information.
		cmd = String.format("perl scorer.pl %s %s %s none", metric,
				conllTestFileName, testConllPredictedFileName);
		execCommandAndRedirectOutputAndError(cmd, new File(scriptBasePath,
				"source/scorer/v4"));
	}

	/**
	 * Execute the given system command and redirects its standard and error
	 * outputs to the standard and error outputs of the JVM process.
	 * 
	 * @param command
	 * @param path
	 * @throws IOException
	 * @throws CommandException
	 * @throws InterruptedException
	 */
	private static void execCommandAndRedirectOutputAndError(String command,
			File path) throws IOException, CommandException,
			InterruptedException {
		String line;
		// Execute command.
		Process p = Runtime.getRuntime().exec(command, null, path);

		// Redirect standard output of process.
		BufferedReader out = new BufferedReader(new InputStreamReader(
				p.getInputStream()));
		while ((line = out.readLine()) != null)
			System.out.println(line);
		out.close();

		// Redirect error output of process.
		BufferedReader error = new BufferedReader(new InputStreamReader(
				p.getErrorStream()));
		while ((line = error.readLine()) != null)
			System.err.println(line);
		error.close();

		if (p.waitFor() != 0)
			throw new CommandException("Command exit with non-zero status");
	}

	private static class CommandException extends Exception {
		/**
		 * Auto-generated serial version ID.
		 */
		private static final long serialVersionUID = 6582860853130630178L;

		public CommandException(String message) {
			super(message);
		}
	}

	/**
	 * Training listener to evaluate models after each iteration.
	 * 
	 * @author eraldof
	 * 
	 */
	private static class EvaluateModelListener implements TrainingListener {

		private String testPredictedFileName;

		private File conllBasePath;

		private String conllTestFileName;

		private String metric;

		private DPColumnDataset testset;

		private DPOutput[] predicteds;

		private boolean averageWeights;

		public EvaluateModelListener(File conllBasePath,
				String testPredictedFileName, String conllTestFileName,
				String metric, CorefColumnDataset testset,
				boolean averageWeights) {
			this.conllBasePath = conllBasePath;
			this.testPredictedFileName = testPredictedFileName;
			this.conllTestFileName = conllTestFileName;
			this.metric = metric;
			this.testset = testset;
			this.averageWeights = averageWeights;
			int numExs = testset.getNumberOfExamples();
			this.predicteds = new DPOutput[numExs];
			// Allocate output sequences for predictions.
			DPInput[] inputs = testset.getInputs();
			for (int idx = 0; idx < numExs; ++idx)
				predicteds[idx] = (DPOutput) inputs[idx].createOutput();
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

			if (averageWeights) {
				try {
					// Clone the current model to average it, if necessary.
					LOG.info("Cloning current model...");
					model = (Model) model.clone();
				} catch (CloneNotSupportedException e) {
					LOG.error("Cloning current model on epoch " + epoch
							+ " and iteration " + iteration, e);
					return true;
				}

				/*
				 * The averaged perceptron averages the final model only in the
				 * end of the training process, hence we need to average the
				 * temporary model here in order to have a better picture of its
				 * current (intermediary) performance.
				 */
				model.average(iteration);
			}

			// Fill the list of predicted outputs.
			DPInput[] inputs = testset.getInputs();
			for (int idx = 0; idx < inputs.length; ++idx)
				// Predict (tag the output sequence).
				inferenceImpl.inference(model, inputs[idx], predicteds[idx]);

			try {
				String testPredictedOnEpochFileName = testPredictedFileName
						+ ".epoch" + epoch;
				LOG.info("Saving test file (" + testPredictedFileName
						+ ") with predicted column...");
				testset.save(testPredictedOnEpochFileName, predicteds);

				try {
					LOG.info("Evaluation after epoch " + epoch + ":");
					// Execute CoNLL evaluation scripts.
					evaluateWithConllScripts(conllBasePath,
							testPredictedOnEpochFileName, conllTestFileName,
							metric);
				} catch (Exception e) {
					LOG.error("Running evaluation scripts", e);
				}
			} catch (IOException e) {
				LOG.error("Saving test file with predicted column", e);
			} catch (DatasetException e) {
				LOG.error("Saving test file with predicted column", e);
			}

			return true;
		}

		@Override
		public void progressReport(Inference impl, Model curModel, int epoch,
				double loss, int iteration) {
		}
	}
}
