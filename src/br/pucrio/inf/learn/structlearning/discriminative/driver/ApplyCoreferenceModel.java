package br.pucrio.inf.learn.structlearning.discriminative.driver;

import java.io.File;
import java.io.IOException;
import java.util.Collection;
import java.util.Random;

import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.OptionBuilder;
import org.apache.commons.cli.Options;
import org.apache.commons.cli.ParseException;
import org.apache.commons.cli.PosixParser;
import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.json.JSONException;

import br.pucrio.inf.learn.structlearning.discriminative.application.coreference.CorefColumnDataset;
import br.pucrio.inf.learn.structlearning.discriminative.application.coreference.CoreferenceMaxBranchInference;
import br.pucrio.inf.learn.structlearning.discriminative.application.coreference.CoreferenceMaxBranchInference.InferenceStrategy;
import br.pucrio.inf.learn.structlearning.discriminative.application.dp.DPModel;
import br.pucrio.inf.learn.structlearning.discriminative.application.dp.DPTemplateEvolutionModel;
import br.pucrio.inf.learn.structlearning.discriminative.application.dp.data.DPInput;
import br.pucrio.inf.learn.structlearning.discriminative.application.dp.data.DPOutput;
import br.pucrio.inf.learn.structlearning.discriminative.data.DatasetException;
import br.pucrio.inf.learn.structlearning.discriminative.data.ExampleInput;
import br.pucrio.inf.learn.structlearning.discriminative.data.ExampleInputArray;
import br.pucrio.inf.learn.structlearning.discriminative.data.encoding.FeatureEncoding;
import br.pucrio.inf.learn.structlearning.discriminative.data.encoding.Murmur3Encoding;
import br.pucrio.inf.learn.structlearning.discriminative.data.encoding.StringMapEncoding;
import br.pucrio.inf.learn.structlearning.discriminative.driver.Driver.Command;
import br.pucrio.inf.learn.util.CommandLineOptionsUtil;

/**
 * Driver to apply a coreference model to a given corpus and, optionally,
 * evaluating the result.
 * 
 * @author eraldo
 * 
 */
public class ApplyCoreferenceModel implements Command {

	/**
	 * Logging object.
	 */
	private static final Log LOG = LogFactory
			.getLog(ApplyCoreferenceModel.class);

	@SuppressWarnings("static-access")
	@Override
	public void run(String[] args) {
		Options options = new Options();
		options.addOption(OptionBuilder.withLongOpt("model")
				.withArgName("filename").hasArg().isRequired()
				.withDescription("File name with the model.").create());
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
		options.addOption(OptionBuilder.withLongOpt("test").isRequired()
				.hasArg().withArgName("filename")
				.withDescription("Test dataset file name.").create());
		options.addOption(OptionBuilder.withLongOpt("scriptpath")
				.withArgName("path").hasArg()
				.withDescription("Base path for CoNLL and Python scripts.")
				.create());
		options.addOption(OptionBuilder.withLongOpt("conlltest")
				.withArgName("filename").hasArg()
				.withDescription("Test dataset on CoNLL format.").create());
		options.addOption(OptionBuilder.withLongOpt("output")
				.withArgName("filename").hasArg()
				.withDescription("Output edge (mention pairs) dataset.")
				.create());
		options.addOption(OptionBuilder.withLongOpt("outputconll")
				.withArgName("filename").hasArg()
				.withDescription("Output CoNLL-format dataset.").create());
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
				.withLongOpt("nosingletons")
				.withDescription("Remove singleton mentions before evaluation.")
				.create());
		options.addOption(OptionBuilder
				.withLongOpt("trees")
				.withDescription(
						"Output edge dataset will include only predicted "
								+ "trees and not all intra-cluster edges.")
				.create());
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
								+ "LKRUSKAL: latent unrirected MS, i.e., "
								+ "use Kruskal algorithm to predict the latent "
								+ "structures. The correct output structures in "
								+ "training data indicate only the correct "
								+ "clusters. The underlying trees are assumed "
								+ "to be latent and are predicted using Kruskal "
								+ "algorithm.").create());

		System.out.println();

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
		String modelFileName = cmdLine.getOptionValue("model");
		String hashSizeStr = cmdLine.getOptionValue("hashsize");
		String hashSeedStr = cmdLine.getOptionValue("hashseed");
		String testDatasetFileName = cmdLine.getOptionValue("test");
		String scriptBasePathStr = cmdLine.getOptionValue("scriptpath");
		String conllTestFileName = cmdLine.getOptionValue("conlltest");
		String outputFileName = cmdLine.getOptionValue("output");
		String outputConllFileName = cmdLine.getOptionValue("outputconll");
		String metric = cmdLine.getOptionValue("conllmetric");
		boolean considerSingletons = !cmdLine.hasOption("nosingletons");
		boolean outputCorefTrees = cmdLine.hasOption("trees");
		// Inference strategy.
		InferenceStrategy inferenceStrategy = InferenceStrategy.BRANCH;
		String inferenceStrategyStr = cmdLine.getOptionValue("inference");
		if (inferenceStrategyStr != null)
			inferenceStrategy = InferenceStrategy.valueOf(inferenceStrategyStr);

		if (outputFileName == null && outputConllFileName == null
				&& metric == null) {
			LOG.error("At least one of --output, --outputconll or --conllmetric must be provided");
			System.exit(1);
		}

		if (outputCorefTrees && (outputFileName == null)) {
			LOG.error("Option --tree requires option --output=<filename>");
			System.exit(1);
		}

		if ((metric == null) != (conllTestFileName == null)) {
			LOG.error("if --conllmetric is provided, then --conlltest"
					+ " must be provided (and vice-versa)");
			System.exit(1);
		}

		// CoNLL scripts base path.
		File conllBasePath = null;
		if (scriptBasePathStr != null)
			conllBasePath = new File(scriptBasePathStr);

		CorefColumnDataset testDataset = null;
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

			LOG.info("Loading dataset...");
			testDataset = new CorefColumnDataset(featureEncoding,
					(Collection<String>) null);
			testDataset.setCheckMultipleTrueEdges(false);
			testDataset.load(testDatasetFileName);

		} catch (Exception e) {
			LOG.error("Parsing command-line options", e);
			System.exit(1);
		}

		LOG.info("Loading model and templates...");
		DPModel model = null;
		try {
			model = new DPTemplateEvolutionModel(modelFileName, testDataset,
					true);
		} catch (JSONException e) {
			LOG.error("Loading model", e);
			System.exit(1);
		} catch (IOException e) {
			LOG.error("Loading model", e);
			System.exit(1);
		} catch (DatasetException e) {
			LOG.error("Loading model", e);
			System.exit(1);
		}

		// LOG.info("Generating features from templates...");
		// testDataset.generateFeatures();

		// Inference algorithm.
		CoreferenceMaxBranchInference inference = new CoreferenceMaxBranchInference(
				testDataset.getMaxNumberOfTokens(), 0, inferenceStrategy);

		/*
		 * Model application.
		 */
		ExampleInputArray inputs = testDataset.getInputs();

		// Allocate predicted output structures.
		DPOutput[] predicteds = new DPOutput[inputs.getNumberExamples()];
		
		inputs.loadInOrder();
		
		for (int idx = 0; idx < inputs.getNumberExamples(); ++idx)
			predicteds[idx] = (DPOutput) inputs.get(idx).createOutput();

		// Fill the list of predicted outputs with predictions from the model.
		LOG.info("Predicting...");
		
		inputs.loadInOrder();
		
		for (int idx = 0; idx < inputs.getNumberExamples(); ++idx) {
			DPInput input = (DPInput) inputs.get(idx);
			// Allocate derived feature matrix memory.
			input.allocFeatureMatrix();
			// Generate derived features from templates.
			input.generateFeatures(testDataset.getTemplates()[0],
					testDataset.getExplicitFeatureEncoding());
			// Predict (tag the output sequence).
			inference.inference(model, input, predicteds[idx]);

			if (!outputCorefTrees)
				// Free derived feature matrix.
				input.freeFeatureMatrix();

			if ((idx + 1) % 100 == 0) {
				System.out.print(".");
				System.out.flush();
			}
		}

		// Predicted test set filename.
		String testPredictedFileName = outputFileName;
		File f = new File(new File(testDatasetFileName).getName());
		if (testPredictedFileName == null)
			testPredictedFileName = f.getAbsolutePath() + "."
					+ new Random().nextInt() + ".pred";

		try {
			if (outputCorefTrees) {
				LOG.info("Predicting \"golden\" trees for the correct outputs...");
				DPOutput[] outputs = testDataset.getOutputs();
				
				inputs.loadInOrder();
				
				for (int idx = 0; idx < inputs.getNumberExamples(); ++idx) {
					DPInput input = (DPInput) inputs.get(idx);
					
					// Constrained prediction.
					inference.partialInference(model, input,
							outputs[idx], outputs[idx]);
					// Free derived feature matrix.
					input.freeFeatureMatrix();
					if ((idx + 1) % 100 == 0) {
						System.out.print(".");
						System.out.flush();
					}
				}
				LOG.info("Saving test file (" + testPredictedFileName
						+ ") with predicted column where correct edges "
						+ "are only the ones in coreference trees...");
				testDataset.saveCorefTrees(testPredictedFileName, predicteds,
						0, true);
			} else {
				LOG.info("Saving test file (" + testPredictedFileName
						+ ") with predicted column...");
				testDataset.save(testPredictedFileName, predicteds);
			}
		} catch (Exception e) {
			LOG.error("Saving predicted file " + testPredictedFileName, e);
			System.exit(1);
		}

		if (metric != null) {
			try {
				LOG.info("Evaluating model...");
				TrainCoreference.evaluateWithConllScripts(conllBasePath,
						testPredictedFileName, conllTestFileName, metric,
						outputConllFileName, considerSingletons);
			} catch (Exception e) {
				LOG.error("Running evaluation scripts", e);
				System.exit(1);
			}
		}

		if (outputFileName == null)
			// Remove temporary predicted file with mention pairs.
			new File(testPredictedFileName).delete();

		LOG.info("Model application done!");
	}

}
