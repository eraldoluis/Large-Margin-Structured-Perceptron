package br.pucrio.inf.learn.structlearning.discriminative.driver;

import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.OptionBuilder;
import org.apache.commons.cli.Options;
import org.apache.commons.cli.ParseException;
import org.apache.commons.cli.PosixParser;

import br.pucrio.inf.learn.structlearning.discriminative.driver.Driver.Command;
import br.pucrio.inf.learn.util.CommandLineOptionsUtil;

public class TrainPQ implements Command {

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
		String additionalCorpusFileName = cmdLine.getOptionValue("inadd");
		String modelFileName = cmdLine.getOptionValue("model");
		int numEpochs = Integer.parseInt(cmdLine.getOptionValue("numepochs",
				"10"));
	}

}
