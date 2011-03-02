package br.pucrio.inf.learn.structlearning.driver;

import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.PrintStream;

import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.HelpFormatter;
import org.apache.commons.cli.Option;
import org.apache.commons.cli.OptionBuilder;
import org.apache.commons.cli.Options;
import org.apache.commons.cli.ParseException;
import org.apache.commons.cli.PosixParser;
import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;

import br.pucrio.inf.learn.structlearning.algorithm.Perceptron;
import br.pucrio.inf.learn.structlearning.application.sequence.ArrayBasedHmm;
import br.pucrio.inf.learn.structlearning.application.sequence.SequenceInput;
import br.pucrio.inf.learn.structlearning.application.sequence.SequenceOutput;
import br.pucrio.inf.learn.structlearning.application.sequence.data.Dataset;
import br.pucrio.inf.learn.structlearning.application.sequence.data.DatasetException;
import br.pucrio.inf.learn.structlearning.data.ExampleOutput;
import br.pucrio.inf.learn.structlearning.data.StringEncoding;
import br.pucrio.inf.learn.structlearning.task.Model;

public class TrainHmmMain implements Driver.Command {

	private static final Log LOG = LogFactory.getLog(TrainHmmMain.class);

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
		options.addOption(OptionBuilder
				.withLongOpt("testoutput")
				.withArgName("test output filename")
				.hasArg()
				.withDescription(
						"File name to save the predicted tags for the test corpus.")
				.create());
		options.addOption(OptionBuilder.withLongOpt("numiter")
				.withArgName("number iterations").hasArg()
				.withDescription("Number of iterations over the training set.")
				.create('T'));
		options.addOption(OptionBuilder
				.withLongOpt("defstate")
				.withArgName("state label")
				.hasArg()
				.withDescription(
						"Default state label to use when all states weight the same.")
				.create('d'));
		options.addOption(OptionBuilder
				.withLongOpt("labels")
				.withArgName("state labels")
				.hasArg()
				.withDescription(
						"List of state labels separated by commas. "
								+ "This can be usefull to specify the preference order of state labels. "
								+ "This option overwrite the following 'tagset' option.")
				.create());
		options.addOption(OptionBuilder
				.withLongOpt("tagset")
				.withArgName("tagset file name")
				.hasArg()
				.withDescription(
						"Name of a file that contains the list of labels, one per line. "
								+ "This can be usefull to specify the preference order of state labels.")
				.create());
		options.addOption(OptionBuilder
				.withLongOpt("verbose")
				.withDescription(
						"Print some information about the execution process.")
				.create('v'));

		// Parse the command-line arguments.
		CommandLine cmdLine = null;
		PosixParser parser = new PosixParser();
		try {
			cmdLine = parser.parse(options, args);
		} catch (ParseException e) {
			usage(options);
		}

		// printOptionValues(cmdLine, options);

		// Get the options specified in the command-line.
		String inputCorpusFileName = cmdLine.getOptionValue('i');
		String testCorpusFileName = cmdLine.getOptionValue('t');
		String testOutputFileName = cmdLine.getOptionValue("testoutput");
		int numIterations = Integer.parseInt(cmdLine.getOptionValue('T', "10"));
		String defaultLabel = cmdLine.getOptionValue('d', "0");
		// boolean verbose = cmdLine.hasOption('v'); // TODO change logging
		// level
		String labels = cmdLine.getOptionValue("labels");
		String tagsetFileName = cmdLine.getOptionValue("tagset");

		Dataset inputCorpus = null;
		try {
			LOG.info("Loading input corpus: " + inputCorpusFileName + "...");
			if (labels != null)
				inputCorpus = new Dataset(inputCorpusFileName,
						new StringEncoding(), new StringEncoding(
								labels.split(",")));
			else if (tagsetFileName != null)
				inputCorpus = new Dataset(inputCorpusFileName,
						new StringEncoding(),
						new StringEncoding(tagsetFileName));
			else
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
						.getStateEncoding().put(defaultLabel));

		LOG.info("Training model...");
		Perceptron alg = new Perceptron(hmm);
		alg.setNumberOfIterations(numIterations);
		alg.train(inputCorpus.getInputs(), inputCorpus.getOutputs(),
				inputCorpus.getFeatureEncoding(),
				inputCorpus.getStateEncoding());

		LOG.info("Saving final model...");
		PrintStream ps;
		try {
			ps = new PrintStream("model.simple");
			alg.getModel().save(ps, inputCorpus.getFeatureEncoding(),
					inputCorpus.getStateEncoding());
			ps.close();
		} catch (FileNotFoundException e) {
			LOG.error("Saving model", e);
		}

		LOG.info("Training done!");

		if (testCorpusFileName != null && testOutputFileName != null) {
			// Test model and save the predicted tags to a file.
			LOG.info("Testing model...");

			// Ignore features not seen in the training corpus.
			inputCorpus.getFeatureEncoding().setReadOnly(true);
			inputCorpus.getStateEncoding().setReadOnly(true);

			try {
				Dataset testCorpus = new Dataset(testCorpusFileName,
						inputCorpus.getFeatureEncoding(),
						inputCorpus.getStateEncoding());

				Model model = alg.getModel();

				PrintStream out = new PrintStream(testOutputFileName);

				int idx = 0;
				for (SequenceInput input : testCorpus.getInputs()) {
					SequenceOutput output = (SequenceOutput) input
							.createOutput();
					// Predict.
					model.inference(input, output);
					// Print predicted tags.
					out.print(idx);
					for (int tkn = 0; tkn < output.size(); ++tkn)
						out.print("\t"
								+ inputCorpus.getStateEncoding()
										.getValueByCode(output.getLabel(tkn)));
					out.println();
					++idx;
				}

				out.close();

			} catch (IOException e) {
				LOG.error("Loading test corpus " + testCorpusFileName, e);
			} catch (DatasetException e) {
				LOG.error("Loading test corpus " + testCorpusFileName, e);
			}
		}
	}

	private void printOptionValues(CommandLine cmdLine, Options options) {
		for (Object obj : options.getOptions()) {
			Option op = (Option) obj;
			String value = ":";
			String[] values = op.getValues();
			if (values != null)
				for (String val : op.getValues())
					value += " " + val;
			System.out.println("\t" + op.getOpt() + "(" + op.getLongOpt() + ")"
					+ value);
		}
	}

	private void usage(Options ops) {
		new HelpFormatter().printHelp("TrainHmm", ops);
		System.exit(1);
	}
}
