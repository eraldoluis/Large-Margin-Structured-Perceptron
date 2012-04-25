package br.pucrio.inf.learn.structlearning.discriminative.driver;

import java.io.File;
import java.io.IOException;
import java.util.Collection;

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
import br.pucrio.inf.learn.structlearning.discriminative.application.dp.DPModel;
import br.pucrio.inf.learn.structlearning.discriminative.application.dp.DPTemplateEvolutionModel;
import br.pucrio.inf.learn.structlearning.discriminative.application.dp.data.DPInput;
import br.pucrio.inf.learn.structlearning.discriminative.application.dp.data.DPOutput;
import br.pucrio.inf.learn.structlearning.discriminative.data.encoding.FeatureEncoding;
import br.pucrio.inf.learn.structlearning.discriminative.data.encoding.StringMapEncoding;
import br.pucrio.inf.learn.structlearning.discriminative.driver.Driver.Command;
import br.pucrio.inf.learn.structlearning.discriminative.task.Inference;
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
		String testDatasetFileName = cmdLine.getOptionValue("test");
		String scriptBasePathStr = cmdLine.getOptionValue("scriptpath");
		String conllTestFileName = cmdLine.getOptionValue("conlltest");
		String outputFileName = cmdLine.getOptionValue("output");
		String metric = cmdLine.getOptionValue("conllmetric");

		if (outputFileName == null && metric == null) {
			LOG.error("At least one of --output and --conllmetric must be provided");
			System.exit(1);
		}

		// CoNLL scripts base path.
		File conllBasePath = null;
		if (scriptBasePathStr != null)
			conllBasePath = new File(scriptBasePathStr);

		/*
		 * If --conllmetric is provided, then --conlltest must be provided (and
		 * vice-versa).
		 */
		if ((metric == null) != (conllTestFileName == null)) {
			LOG.error("if --conllmetric is provided, then --conlltest"
					+ " must be provided (and vice-versa)");
			System.exit(1);
		}

		CorefColumnDataset testDataset = null;
		FeatureEncoding<String> featureEncoding = null;
		try {
			/*
			 * Create an empty and flexible feature encoding that will encode
			 * unambiguously all feature values. If the training dataset is big,
			 * this may not fit in memory.
			 */
			featureEncoding = new StringMapEncoding();

			LOG.info("Loading train dataset...");
			testDataset = new CorefColumnDataset(featureEncoding,
					(Collection<String>) null);
			((CorefColumnDataset) testDataset).setCheckMultipleTrueEdges(false);
			testDataset.load(testDatasetFileName);

		} catch (Exception e) {
			LOG.error("Parsing command-line options", e);
			System.exit(1);
		}

		LOG.info("Loading model and templates...");
		DPModel model = null;
		try {
			model = new DPTemplateEvolutionModel(modelFileName, testDataset);
		} catch (JSONException e) {
			LOG.error("Loading model", e);
			System.exit(1);
		} catch (IOException e) {
			LOG.error("Loading model", e);
			System.exit(1);
		}

		LOG.info("Generating features from templates...");
		testDataset.generateFeatures();

		// Inference algorithm.
		Inference inference = new CoreferenceMaxBranchInference(
				testDataset.getMaxNumberOfTokens(), 0);

		/*
		 * Model application.
		 */
		DPInput[] inputs = testDataset.getInputs();

		// Allocate predicted output structures.
		DPOutput[] predicteds = new DPOutput[inputs.length];
		for (int idx = 0; idx < inputs.length; ++idx)
			predicteds[idx] = inputs[idx].createOutput();

		// Fill the list of predicted outputs with predictions from the model.
		for (int idx = 0; idx < inputs.length; ++idx)
			// Predict (tag the output sequence).
			inference.inference(model, inputs[idx], predicteds[idx]);

		// Predicted test set filename.
		String testPredictedFileName = outputFileName;
		if (testPredictedFileName == null)
			testPredictedFileName = testDatasetFileName + ".pred";

		try {
			LOG.info("Saving test file (" + testPredictedFileName
					+ ") with predicted column...");
			testDataset.save(testPredictedFileName, predicteds);
		} catch (Exception e) {
			LOG.error("Saving predicted file " + testPredictedFileName, e);
			System.exit(1);
		}

		if (metric != null) {
			try {
				LOG.info("Evaluating model...");
				TrainCoreference.evaluateWithConllScripts(conllBasePath,
						testPredictedFileName, conllTestFileName, metric);
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
