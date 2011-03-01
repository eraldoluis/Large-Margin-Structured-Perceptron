package br.pucrio.inf.learn.structlearning.driver;

import java.io.IOException;

import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.HelpFormatter;
import org.apache.commons.cli.OptionBuilder;
import org.apache.commons.cli.Options;
import org.apache.commons.cli.ParseException;
import org.apache.commons.cli.PosixParser;
import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;

import br.pucrio.inf.learn.structlearning.algorithm.Perceptron;
import br.pucrio.inf.learn.structlearning.application.sequence.ArrayBasedHmm;
import br.pucrio.inf.learn.structlearning.application.sequence.data.Dataset;
import br.pucrio.inf.learn.structlearning.application.sequence.data.DatasetException;

public class TrainHmmMain implements Driver.Command {

	private static final Log LOG = LogFactory.getLog(TrainHmmMain.class);

	@SuppressWarnings("static-access")
	@Override
	public void run(String[] args) {
		Options options = new Options();
		options.addOption(OptionBuilder.withLongOpt("incorpus").isRequired()
				.withArgName("input corpus").hasArg()
				.withDescription("Input corpus file name").create('i'));
		options.addOption(OptionBuilder
				.withLongOpt("verbose")
				.withDescription(
						"Print some information about the execution process")
				.create('v'));

		CommandLine cmdLine = null;
		PosixParser parser = new PosixParser();
		try {
			cmdLine = parser.parse(options, args);
		} catch (ParseException e) {
			LOG.error("Parsing command-line arguments", e);
			usage(options);
		}

		String inputCorpusFileName = cmdLine.getOptionValue('i');
		if (cmdLine.hasOption('v'))
			; // TODO change log-level

		Dataset inputCorpus = null;
		try {
			LOG.info("Loading input corpus: " + inputCorpusFileName + "...");
			inputCorpus = new Dataset(inputCorpusFileName);
		} catch (IOException e) {
			LOG.error("Loading input corpus", e);
			System.exit(1);
		} catch (DatasetException e) {
			LOG.error("Loading input corpus", e);
			System.exit(1);
		}

		LOG.info("Allocating initial model...");
		ArrayBasedHmm hmm = new ArrayBasedHmm(inputCorpus.getNumberOfStates(),
				inputCorpus.getNumberOfSymbols(), inputCorpus
						.getStateEncoding().putValue("0"));

		LOG.info("Training model...");
		Perceptron alg = new Perceptron(hmm);
		alg.train(inputCorpus.getInputs(), inputCorpus.getOutputs());
	}

	private void usage(Options ops) {
		new HelpFormatter().printHelp("TrainHmm", ops);
	}
}
