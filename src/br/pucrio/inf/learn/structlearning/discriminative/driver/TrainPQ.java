package br.pucrio.inf.learn.structlearning.discriminative.driver;

import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.OptionBuilder;
import org.apache.commons.cli.Options;
import org.apache.commons.cli.ParseException;
import org.apache.commons.cli.PosixParser;
import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;

import br.pucrio.inf.learn.structlearning.discriminative.application.pq.data.PQDataset;
import br.pucrio.inf.learn.structlearning.discriminative.data.encoding.FeatureEncoding;
import br.pucrio.inf.learn.structlearning.discriminative.data.encoding.StringMapEncoding;
import br.pucrio.inf.learn.structlearning.discriminative.driver.Driver.Command;
import br.pucrio.inf.learn.util.CommandLineOptionsUtil;

public class TrainPQ implements Command {

	/**
	 * Logging object.
	 */
	private static final Log LOG = LogFactory.getLog(TrainHmm.class);

	@SuppressWarnings("static-access")
	@Override
	public void run(String[] args) {
		Options options = new Options();
		options.addOption(OptionBuilder
				.withLongOpt("task")
				.withArgName("iob | token")
				.hasArg()
				.isRequired()
				.withDescription(
						"Which type of task is performed: IOB sequence "
								+ "labeling or token labeling").create());
		options.addOption(OptionBuilder.withLongOpt("incorpus").isRequired()
				.withArgName("input corpus").hasArg()
				.withDescription("Input corpus file name.").create('i'));

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
	}
}
