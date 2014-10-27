package br.pucrio.inf.learn.structlearning.discriminative.driver;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.Collection;
import java.util.Random;

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
import br.pucrio.inf.learn.structlearning.discriminative.application.coreference.CorefColumnDataset;
import br.pucrio.inf.learn.structlearning.discriminative.application.coreference.CorefInput;
import br.pucrio.inf.learn.structlearning.discriminative.application.coreference.CorefModel;
import br.pucrio.inf.learn.structlearning.discriminative.application.coreference.CorefModel.UpdateStrategy;
import br.pucrio.inf.learn.structlearning.discriminative.application.coreference.CorefOutput;
import br.pucrio.inf.learn.structlearning.discriminative.application.coreference.CorefUndirectedModel;
import br.pucrio.inf.learn.structlearning.discriminative.application.coreference.CoreferenceMaxBranchInference;
import br.pucrio.inf.learn.structlearning.discriminative.application.coreference.CoreferenceMaxBranchInference.InferenceStrategy;
import br.pucrio.inf.learn.structlearning.discriminative.application.dp.DPModel;
import br.pucrio.inf.learn.structlearning.discriminative.application.dp.DPTemplateEvolutionModel;
import br.pucrio.inf.learn.structlearning.discriminative.application.dp.MaximumBranchingInference;
import br.pucrio.inf.learn.structlearning.discriminative.application.dp.data.DPColumnDataset;
import br.pucrio.inf.learn.structlearning.discriminative.application.dp.data.DPOutput;
import br.pucrio.inf.learn.structlearning.discriminative.data.DatasetException;
import br.pucrio.inf.learn.structlearning.discriminative.data.ExampleInputArray;
import br.pucrio.inf.learn.structlearning.discriminative.data.encoding.FeatureEncoding;
import br.pucrio.inf.learn.structlearning.discriminative.data.encoding.Murmur3Encoding;
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
		options.addOption(OptionBuilder
				.withLongOpt("inference")
				.withArgName("inference strategy")
				.hasArg()
				.withDescription(
						"Inference strategy and algorithm. "
								+ "It can be one of the following options:\n"
								+ "BRANCH: fixed maximum branching. The "
								+ "correct output structures in training "
								+ "data indicate the correct tree.\n"
								+ "LBRANCH: latent maximum branching. The "
								+ "correct output structures in training data "
								+ "indicate only the correct clusters. The "
								+ "underlying tress are latent.\n"
								+ "LKRUSKAL: latent undirected MST, i.e., "
								+ "use Kruskal algorithm to predict the latent "
								+ "structures. The correct output structures in "
								+ "training data indicate only the correct "
								+ "clusters. The underlying trees are assumed "
								+ "to be latent and are predicted using Kruskal "
								+ "algorithm.\n"
								+ "CHAIN: During training, do not use "
								+ "the current model to predict the golden "
								+ "coreference trees. It chooses the closest "
								+ " (previous) mention as the parent mention. "
								+ "Usually, this rule derives a chain (sequential) "
								+ "of mentions for each entity cluster. However, "
								+ "since the input dataset may not include all "
								+ "edges, for some cases, a completely sequential "
								+ "chain is not possible and the resulting "
								+ "structure will be a tree.").create());
		options.addOption(OptionBuilder
				.withLongOpt("update")
				.withArgName("strategy")
				.hasArg()
				.withDescription(
						"Update strategy to train: "
								+ "CLUSTER (default), TREE or ALL.").create());
		options.addOption(OptionBuilder
				.withLongOpt("noroot")
				.withDescription(
						"Do not use the artificial root node."
								+ " In that way, the prediction algorithm "
								+ "avoids negative-weight edges in order to "
								+ "predict non-connected trees and, "
								+ "consequently, more than one cluster.")
				.create());
		options.addOption(OptionBuilder
				.withLongOpt("rootlossfactor")
				.withArgName("multiplicative factor")
				.hasArg()
				.withDescription(
						"Multiplicative factor to be used for root edges.")
				.create());
		options.addOption(OptionBuilder.withLongOpt("templates").isRequired()
				.withArgName("filename").hasArg()
				.withDescription("Feature templates file name.").create());
		options.addOption(OptionBuilder
				.withLongOpt("hashsize")
				.withArgName("size")
				.hasArg()
				.withDescription(
						"Number of entries or bits (use "
								+ "suffix b) in the hash function.").create());
		options.addOption(OptionBuilder.withLongOpt("hashseed")
				.withArgName("seed").hasArg()
				.withDescription("Seed for the hash function.").create());
		options.addOption(OptionBuilder
				.withLongOpt("numepochs")
				.withArgName("integer")
				.hasArg()
				.withDescription(
						"Number of epochs: how many iterations over the"
								+ " training set.").create());
		options.addOption(OptionBuilder.withLongOpt("model")
				.withArgName("filename").hasArg()
				.withDescription("File name to save final model.").create());
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
						"Evaluation metric (use comma to separate multiple values):\n"
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
		options.addOption(OptionBuilder.withLongOpt("nosingletons")
				.withDescription("Remove singleton metions before evaluation.")
				.create());

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
		String modelFileName = cmdLine.getOptionValue("model");
		String hashSizeStr = cmdLine.getOptionValue("hashsize");
		String hashSeedStr = cmdLine.getOptionValue("hashseed");
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
		// Inference strategy.
		InferenceStrategy inferenceStrategy = InferenceStrategy.BRANCH;
		String inferenceStrategyStr = cmdLine.getOptionValue("inference");
		if (inferenceStrategyStr != null)
			inferenceStrategy = InferenceStrategy.valueOf(inferenceStrategyStr);
		// Update strategy.
		UpdateStrategy updateStrategy = null;
		String updateStrategyStr = cmdLine.getOptionValue("update");
		if (updateStrategyStr != null)
			updateStrategy = UpdateStrategy.valueOf(updateStrategyStr);
		boolean useRoot = !cmdLine.hasOption("noroot");
		// Root loss factor.
		double rootLossFactor = Double.valueOf(cmdLine.getOptionValue(
				"rootlossfactor", "-1"));
		boolean considerSingletons = !cmdLine.hasOption("nosingletons");

		/*
		 * If --test is provided, then --conlltest must be provided (and
		 * vice-versa).
		 */
		if ((testDatasetFileName == null) != (conllTestFileName == null)) {
			LOG.error("if --test is provided, then --conlltest"
					+ " must be provided (and vice-versa)");
			System.exit(1);
		}

		DPColumnDataset inDataset = null;
		FeatureEncoding<String> featureEncoding = null;
		try {
			/*
			 * The feature encoding converts textual feature values to integer
			 * codes. The use has two options: simple encoding and hash
			 * encoding. The simple encoding only gives a new code to each
			 * unseen feature value. The hash encoding uses a hash function
			 * (Murmur 3, in our case) to convert any string value to a code.
			 * The latter case implies on some representation loss, due to
			 * collisions in the hash function. On the other hand, the hash
			 * trick can save lots of memory.
			 */
			if (hashSizeStr == null) {
				if (hashSeedStr == null) {
					featureEncoding = new StringMapEncoding();
				} else {
					LOG.error("Option --hashseed requires --hashsize");
					System.exit(1);
				}
			} else {
				int hashSize = TrainDP.parseValueDirectOrBits(hashSizeStr);
				if (hashSeedStr == null) {
					featureEncoding = new Murmur3Encoding(hashSize);
				} else {
					int hashSeed = Integer.parseInt(hashSeedStr);
					featureEncoding = new Murmur3Encoding(hashSize, hashSeed);
				}
			}

			LOG.info("Loading train dataset...");
			if (inferenceStrategy != InferenceStrategy.BRANCH) {
				// Latent output structure.
				inDataset = new CorefColumnDataset(featureEncoding,
						(Collection<String>) null);
				((CorefColumnDataset) inDataset)
						.setCheckMultipleTrueEdges(false);
			} else {
				// Explicit output structure.
				inDataset = new DPColumnDataset(featureEncoding,
						(Collection<String>) null);
			}

			inDataset.load(inputCorpusFileNames[0]);

			LOG.info("Loading templates and generating features...");
			inDataset.loadTemplates(templatesFileName, true);

			// Generate explicit features from templates.
			// inDataset.generateFeatures();
		} catch (Exception e) {
			LOG.error("Parsing command-line options", e);
			System.exit(1);
		}

		// Template-based model.
		LOG.info("Allocating initial model...");
		DPModel model;

		// Inference (prediction) algorithm.
		Inference inference;

		if (inferenceStrategy == InferenceStrategy.LBRANCH) {
			// Model and the update strategy.
			model = new CorefModel(0);
			if (updateStrategy != null)
				((CorefModel) model).setUpdateStrategy(updateStrategy);
			// Inference (prediction) algorithm and the root loss factor.
			inference = new CoreferenceMaxBranchInference(
					inDataset.getMaxNumberOfTokens(), 0, inferenceStrategy);
			((CoreferenceMaxBranchInference) inference).setUseRoot(useRoot);
			if (rootLossFactor >= 0d)
				((CoreferenceMaxBranchInference) inference)
						.setLossFactorForRootEdges(rootLossFactor);
		} else if (inferenceStrategy == InferenceStrategy.LKRUSKAL) {
			// Model and the update strategy.
			model = new CorefUndirectedModel(0);
			if (updateStrategy != null)
				((CorefModel) model).setUpdateStrategy(updateStrategy);
			// Inference (prediction) algorithm and the root loss factor.
			inference = new CoreferenceMaxBranchInference(
					inDataset.getMaxNumberOfTokens(), 0, inferenceStrategy);
			((CoreferenceMaxBranchInference) inference).setUseRoot(useRoot);
			if (rootLossFactor >= 0d)
				((CoreferenceMaxBranchInference) inference)
						.setLossFactorForRootEdges(rootLossFactor);
		} else if (inferenceStrategy == InferenceStrategy.BRANCH) {
			if (updateStrategy != null) {
				LOG.error("--update=<strategy> requires --inference=LBRANCH");
				System.exit(1);
			}
			if (rootLossFactor >= 0d) {
				LOG.error("--rootlossfactor=<factor> requires --latent");
				System.exit(1);
			}
			model = new DPTemplateEvolutionModel(0);
			inference = new MaximumBranchingInference(
					inDataset.getMaxNumberOfTokens());
		} else if (inferenceStrategy == inferenceStrategy.CHAIN) {
			// Model and the update strategy.
			model = new CorefModel(0);
			if (updateStrategy != null)
				((CorefModel) model).setUpdateStrategy(updateStrategy);

			// Inference (prediction) algorithm and the root loss factor.
			inference = new CoreferenceMaxBranchInference(
					inDataset.getMaxNumberOfTokens(), 0, inferenceStrategy);
			((CoreferenceMaxBranchInference) inference).setUseRoot(useRoot);
			if (rootLossFactor >= 0d)
				((CoreferenceMaxBranchInference) inference)
						.setLossFactorForRootEdges(rootLossFactor);

			LOG.info("Computing chains...");

			/*
			 * Generate golden chains for the documents. In this inference
			 * strategy, we do not use latent structures to represent clusters.
			 * Instead, we use *fixed* chains of mentions to represent each
			 * golden cluster.
			 */
			DPOutput[] outs = ((CorefColumnDataset) inDataset).getOutputs();
			ExampleInputArray ins = ((CorefColumnDataset) inDataset).getInputs();
			int idxDoc = 0;
			
			ins.loadInOrder();
			
			for (DPOutput out : outs) {
				// Current coreference output and input structures.
				CorefOutput cout = (CorefOutput) out;
				CorefInput cin = (CorefInput) ins.get(idxDoc);
				// Number of mentions.
				int numMentions = cout.size();

				/*
				 * For each mention, find the closest (previous) head mention in
				 * the same cluster and with existing edge from the head mention
				 * and the current mention.
				 */
				cout.setHead(0, -1);
				cout.setHead(1, 0);
				for (int idxMention = 2; idxMention < numMentions; ++idxMention) {
					// Start with the artificial mention as the head.
					cout.setHead(idxMention, 0);

					// Cluster id of the current mention.
					int cId = cout.getClusterId(idxMention);

					for (int idxMentionHead = idxMention - 1; idxMentionHead > 0; --idxMentionHead) {
						if (cId == cout.getClusterId(idxMentionHead)
								&& cin.getBasicFeatures(idxMentionHead,
										idxMention) != null) {
							/*
							 * Set the head when find an existing edge
							 * connection to the same cluster.
							 */
							cout.setHead(idxMention, idxMentionHead);
							break;
						}
					}
				}

				++idxDoc;
			}
		} else {
			LOG.error("Unknown inference strategy "
					+ inferenceStrategy.toString());
			System.exit(1);
			return;
		}

		LOG.info("Setting learning algorithm...");

		// Learning algorithm.
		LossAugmentedPerceptron alg = new LossAugmentedPerceptron(inference,
				model, numEpochs, 1d, lossWeight, true, averageWeights,
				LearnRateUpdateStrategy.NONE);

		if (inferenceStrategy == InferenceStrategy.LBRANCH
				|| inferenceStrategy == InferenceStrategy.LKRUSKAL)
			alg.setPartiallyAnnotatedExamples(true);

		if (seedStr != null)
			// User provided seed to random number generator.
			alg.setSeed(Long.parseLong(seedStr));

		if (reportProgressRate != null)
			// Progress report rate.
			alg.setReportProgressRate(reportProgressRate);

		// Ignore features not seen in the training corpus.
		// featureEncoding.setReadOnly(true);

		DPColumnDataset testset = null;

		// Evaluation after each training epoch.
		if (testDatasetFileName != null && evalPerEpoch) {
			try {
				LOG.info("Loading and preparing test dataset...");
				if (inferenceStrategy == InferenceStrategy.LBRANCH
						|| inferenceStrategy == InferenceStrategy.LKRUSKAL
						|| inferenceStrategy == InferenceStrategy.CHAIN) {
					testset = new CorefColumnDataset(inDataset);
					((CorefColumnDataset) testset)
							.setCheckMultipleTrueEdges(false);
				} else {
					testset = new DPColumnDataset(inDataset);
				}
				testset.load(testDatasetFileName);
				LOG.info("Generating features from templates...");
				testset.generateFeatures();
				// Predicted test set filename.
				File f = new File(new File(testDatasetFileName).getName());
				String testPredictedFileName = f.getAbsolutePath() + "."
						+ new Random().nextInt() + ".pred";
				// Set listener that perform evaluation after each epoch.
				alg.setListener(new EvaluateModelListener(conllBasePath,
						testPredictedFileName, conllTestFileName, metric,
						testset, averageWeights, considerSingletons));
			} catch (Exception e) {
				LOG.error("Loading testset " + testDatasetFileName, e);
				System.exit(1);
			}
		}

		LOG.info("Training model...");
		// Train model.
		alg.train(inDataset.getInputs(), inDataset.getOutputs());

		if (modelFileName != null) {
			try {
				LOG.info(String.format("Saving model on file %s...",
						modelFileName));
				model.save(modelFileName, inDataset);
			} catch (FileNotFoundException e) {
				LOG.error(e);
			} catch (IOException e) {
				LOG.error(e);
			}
		}

		// Evaluation only for the final model.
		if (testDatasetFileName != null && !evalPerEpoch) {
			try {

				LOG.info("Loading and preparing test dataset...");
				if (inferenceStrategy == InferenceStrategy.LBRANCH
						|| inferenceStrategy == InferenceStrategy.LKRUSKAL
						|| inferenceStrategy == InferenceStrategy.CHAIN) {
					testset = new CorefColumnDataset(inDataset);
					((CorefColumnDataset) testset)
							.setCheckMultipleTrueEdges(false);
				} else {
					testset = new DPColumnDataset(inDataset);
				}
				testset.load(testDatasetFileName);
				LOG.info("Generating features from templates...");
				testset.generateFeatures();

			} catch (Exception e) {
				LOG.error("Loading testset " + testDatasetFileName, e);
				System.exit(1);
			}

			// Allocate output sequences for predictions.
			ExampleInputArray inputs = testset.getInputs();
			DPOutput[] predicteds = new DPOutput[inputs.getNumberExamples()];
			
			inputs.loadInOrder();
			
			for (int idx = 0; idx < inputs.getNumberExamples(); ++idx)
				predicteds[idx] = (DPOutput) inputs.get(idx).createOutput();

			inputs.loadInOrder();
			
			// Fill the list of predicted outputs.
			for (int idx = 0; idx < inputs.getNumberExamples(); ++idx)
				// Predict (tag the output sequence).
				inference.inference(model, inputs.get(idx), predicteds[idx]);

			// Predicted test set filename.
			File f = new File(new File(testDatasetFileName).getName());
			String testPredictedFileName = f.getAbsolutePath() + "."
					+ new Random().nextInt() + ".pred";

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
						conllTestFileName, metric, null, considerSingletons);
			} catch (Exception e) {
				LOG.error("Running evaluation scripts", e);
				System.exit(1);
			}

			// Remove temporary predicted file with mention pairs.
			new File(testPredictedFileName).delete();
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
	 * @param outputConll
	 * @param considerSingletons
	 * @throws IOException
	 * @throws CommandException
	 * @throws InterruptedException
	 */
	public static void evaluateWithConllScripts(File scriptBasePath,
			String testPredictedFileName, String conllTestFileName,
			String metric, String outputConll, boolean considerSingletons)
			throws IOException, CommandException, InterruptedException {
		// File name of CoNLL-format predicted file.
		String testConllPredictedFileName = outputConll;
		if (testConllPredictedFileName == null)
			testConllPredictedFileName = testPredictedFileName + ".conll";
		// Command to convert the predicted file to CoNLL format.
		String cmd = String.format(
				"python mentionPairsToConll.py %s %s %s predicted",
				conllTestFileName, testPredictedFileName,
				testConllPredictedFileName);
		if (!considerSingletons)
			cmd += " NOSINGLETON";
		execCommandAndRedirectOutputAndError(cmd, scriptBasePath);

		String[] metrics = metric.split(",");
		for (String m : metrics) {
			LOG.info(String.format("*** Results for %s ***", m));
			// Command to evaluate the predicted information.
			cmd = String.format("perl scorer.pl %s %s %s none", m,
					conllTestFileName, testConllPredictedFileName);
			execCommandAndRedirectOutputAndError(cmd, new File(scriptBasePath,
					"scorer/v4"));
		}

		// Remove temporary file.
		if (outputConll == null)
			new File(testConllPredictedFileName).delete();
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
		LOG.info("Running command: " + command);
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

		private boolean considerSingletons;

		public EvaluateModelListener(File conllBasePath,
				String testPredictedFileName, String conllTestFileName,
				String metric, DPColumnDataset testset, boolean averageWeights,
				boolean considerSingletons) {
			this.conllBasePath = conllBasePath;
			this.testPredictedFileName = testPredictedFileName;
			this.conllTestFileName = conllTestFileName;
			this.metric = metric;
			this.testset = testset;
			this.averageWeights = averageWeights;
			this.considerSingletons = considerSingletons;
			int numExs = testset.getNumberOfExamples();
			// Allocate output sequences for predictions.
			ExampleInputArray inputs = testset.getInputs();
			this.predicteds = new DPOutput[numExs];
			
			inputs.loadInOrder();
			
			for (int idx = 0; idx < numExs; ++idx)
				predicteds[idx] = (DPOutput) inputs.get(idx).createOutput();
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
			ExampleInputArray inputs = testset.getInputs();
			// DPOutput[] outputs = testset.getOutputs();
			for (int idx = 0; idx < inputs.getNumberExamples(); ++idx)
				// Predict (tag the output sequence).
				inferenceImpl.inference(model, inputs.get(idx), predicteds[idx]);

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
							metric, null, considerSingletons);
				} catch (Exception e) {
					LOG.error("Running evaluation scripts", e);
				}

				// Delete temporary file.
				new File(testPredictedOnEpochFileName).delete();

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
