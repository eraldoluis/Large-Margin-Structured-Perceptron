package br.pucrio.inf.learn.structlearning.discriminative.driver;

import java.util.Collection;

import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.OptionBuilder;
import org.apache.commons.cli.Options;
import org.apache.commons.cli.ParseException;
import org.apache.commons.cli.PosixParser;
import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;

import br.pucrio.inf.learn.structlearning.discriminative.application.coreference.CorefColumnDataset;
import br.pucrio.inf.learn.structlearning.discriminative.application.dp.data.DPOutput;
import br.pucrio.inf.learn.structlearning.discriminative.data.encoding.FeatureEncoding;
import br.pucrio.inf.learn.structlearning.discriminative.data.encoding.StringMapEncoding;
import br.pucrio.inf.learn.util.CommandLineOptionsUtil;

/**
 * Driver to apply a coreference model to a given corpus and, optionally,
 * evaluating the result.
 * 
 * @author eraldo
 * 
 */
public class TestCorefTree {

	/**
	 * Logging object.
	 */
	private static final Log LOG = LogFactory.getLog(TestCorefTree.class);

	@SuppressWarnings("static-access")
	public static void main(String[] args) {
		Options options = new Options();
		options.addOption(OptionBuilder.withLongOpt("test").isRequired()
				.hasArg().withArgName("filename")
				.withDescription("Test dataset file name.").create());
		options.addOption(OptionBuilder.withLongOpt("output").isRequired()
				.withArgName("filename").hasArg()
				.withDescription("Output edge (mention pairs) dataset.")
				.create());

		System.out.println();

		// Parse the command-line arguments.
		CommandLine cmdLine = null;
		PosixParser parser = new PosixParser();
		try {
			cmdLine = parser.parse(options, args);
		} catch (ParseException e) {
			System.err.println(e.getMessage());
			CommandLineOptionsUtil.usage("TestCorefTree", options);
		}

		// Print the list of options along the values provided by the user.
		CommandLineOptionsUtil.printOptionValues(cmdLine, options);

		/*
		 * Get the options given in the command-line or the corresponding
		 * default values.
		 */
		String testDatasetFileName = cmdLine.getOptionValue("test");
		String outputFileName = cmdLine.getOptionValue("output");

		CorefColumnDataset testDataset = null;
		FeatureEncoding<String> featureEncoding = null;
		try {
			/*
			 * Create an empty and flexible feature encoding that will encode
			 * unambiguously all feature values. If the training dataset is big,
			 * this may not fit in memory.
			 */
			featureEncoding = new StringMapEncoding();

			LOG.info("Loading dataset...");
			testDataset = new CorefColumnDataset(featureEncoding,
					(Collection<String>) null);
			testDataset.setCheckMultipleTrueEdges(false);
			testDataset.load(testDatasetFileName);

		} catch (Exception e) {
			LOG.error("Parsing command-line options", e);
			System.exit(1);
		}

		LOG.info("Generating features from templates...");
		testDataset.generateBasicFeatures();

		DPOutput[] outputs = testDataset.getOutputs();

		try {
			LOG.info("Saving test file (" + outputFileName
					+ ") with predicted column where correct edges "
					+ "are only the ones in coreference trees...");
			testDataset.saveCorefTrees(outputFileName, outputs, 0, false);
		} catch (Exception e) {
			LOG.error("Saving predicted file " + outputFileName, e);
			System.exit(1);
		}

		LOG.info("Test finished!");
	}

}
