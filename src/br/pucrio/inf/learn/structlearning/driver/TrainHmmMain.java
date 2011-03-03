package br.pucrio.inf.learn.structlearning.driver;

import java.io.FileNotFoundException;
import java.io.PrintStream;
import java.util.Map;

import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.OptionBuilder;
import org.apache.commons.cli.Options;
import org.apache.commons.cli.ParseException;
import org.apache.commons.cli.PosixParser;
import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;

import br.pucrio.inf.learn.structlearning.algorithm.Perceptron;
import br.pucrio.inf.learn.structlearning.algorithm.Perceptron.Listener;
import br.pucrio.inf.learn.structlearning.application.sequence.ArrayBasedHmm;
import br.pucrio.inf.learn.structlearning.application.sequence.SequenceInput;
import br.pucrio.inf.learn.structlearning.application.sequence.SequenceOutput;
import br.pucrio.inf.learn.structlearning.application.sequence.data.Dataset;
import br.pucrio.inf.learn.structlearning.application.sequence.evaluation.F1Measure;
import br.pucrio.inf.learn.structlearning.application.sequence.evaluation.IobChunkEvaluation;
import br.pucrio.inf.learn.structlearning.data.StringEncoding;
import br.pucrio.inf.learn.structlearning.task.Model;
import br.pucrio.inf.learn.util.CommandLineOptionsUtil;

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
				.withLongOpt("numepochs")
				.withArgName("number of epochs")
				.hasArg()
				.withDescription(
						"Number of epochs: how many "
								+ "iterations over the training set.")
				.create('T'));
		options.addOption(OptionBuilder.withLongOpt("learnrate")
				.withArgName("learning rate within [0:1]").hasArg()
				.withDescription("Learning rate used in the updates.").create());
		options.addOption(OptionBuilder
				.withLongOpt("defstate")
				.withArgName("state label")
				.hasArg()
				.withDescription(
						"Default state label to use when all states weight the same.")
				.create('d'));
		options.addOption(OptionBuilder
				.withLongOpt("nullstate")
				.withArgName("state label")
				.hasArg()
				.withDescription(
						"Null state label if different of default state.")
				.create());
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
						"Print additional information about the execution process.")
				.create('v'));

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

		// Get the options specified in the command-line.
		String inputCorpusFileName = cmdLine.getOptionValue('i');
		String testCorpusFileName = cmdLine.getOptionValue('t');
		int numEpochs = Integer.parseInt(cmdLine.getOptionValue('T', "10"));
		double learningRate = Double.parseDouble(cmdLine.getOptionValue(
				"learnrate", "1"));
		String defaultLabel = cmdLine.getOptionValue('d', "0");
		String nullLabel = cmdLine.getOptionValue("nullstate", defaultLabel);
		// boolean verbose = cmdLine.hasOption('v');
		String labels = cmdLine.getOptionValue("labels");
		String tagsetFileName = cmdLine.getOptionValue("tagset");

		Dataset inputCorpus = null;
		try {

			LOG.info("Loading input corpus: " + inputCorpusFileName + "...");

			if (labels != null)
				// State set provided explicitly as a command-line argument.
				inputCorpus = new Dataset(inputCorpusFileName,
						new StringEncoding(), new StringEncoding(
								labels.split(",")));
			else if (tagsetFileName != null)
				// State set provided in a tagset file.
				inputCorpus = new Dataset(inputCorpusFileName,
						new StringEncoding(),
						new StringEncoding(tagsetFileName));
			else
				// State set automatically retrieved from training data (codes
				// depend on order of appereance of the labels).
				inputCorpus = new Dataset(inputCorpusFileName);

		} catch (Exception e) {
			LOG.error("Loading input corpus", e);
			System.exit(1);
		}

		LOG.info("Allocating initial model...");
		ArrayBasedHmm hmm = new ArrayBasedHmm(inputCorpus.getNumberOfStates(),
				inputCorpus.getNumberOfSymbols(), inputCorpus
						.getStateEncoding().put(defaultLabel));

		Perceptron alg = new Perceptron(hmm);
		alg.setNumberOfEpochs(numEpochs);
		alg.setLearningRate(learningRate);

		if (testCorpusFileName != null) {
			// Ignore features not seen in the training corpus.
			inputCorpus.getFeatureEncoding().setReadOnly(true);
			inputCorpus.getStateEncoding().setReadOnly(true);

			try {

				LOG.info("Loading and preparing test data...");
				Dataset testset = new Dataset(testCorpusFileName,
						inputCorpus.getFeatureEncoding(),
						inputCorpus.getStateEncoding());
				alg.setListener(new EvaluateModelListener(testset.getInputs(),
						testset.getOutputs(), inputCorpus.getStateEncoding(),
						nullLabel));

			} catch (Exception e) {
				LOG.error("Loading testset " + testCorpusFileName, e);
				System.exit(1);
			}
		}

		LOG.info("Training model...");
		alg.train(inputCorpus.getInputs(), inputCorpus.getOutputs(),
				inputCorpus.getFeatureEncoding(),
				inputCorpus.getStateEncoding());

//		if (testCorpusFileName != null) {
//			try {
//
//				LOG.info("Loading and preparing test data...");
//				Dataset testset = new Dataset(testCorpusFileName,
//						inputCorpus.getFeatureEncoding(),
//						inputCorpus.getStateEncoding());
//
//				// Allocate output sequences for predictions.
//				SequenceInput[] inputs = testset.getInputs();
//				SequenceOutput[] outputs = testset.getOutputs();
//				SequenceOutput[] predicteds = new SequenceOutput[inputs.length];
//				for (int idx = 0; idx < inputs.length; ++idx)
//					predicteds[idx] = (SequenceOutput) inputs[idx]
//							.createOutput();
//				IobChunkEvaluation eval = new IobChunkEvaluation(
//						inputCorpus.getStateEncoding(), nullLabel);
//
//				// Fill the list of predicted outputs.
//				for (int idx = 0; idx < inputs.length; ++idx)
//					// Predict (tag the output sequence).
//					hmm.inference(inputs[idx], predicteds[idx]);
//
//				// Evaluate the sequences.
//				Map<String, F1Measure> results = eval.evaluateSequences(inputs,
//						outputs, predicteds);
//
//				// Write results: precision, recall and F-1 values.
//				System.out.println();
//				System.out.println("|  *Class*  |  *P*  |  *R*  |  *F*  |");
//				String[] labelOrder = { "LOC", "MISC", "ORG", "PER", "overall" };
//				for (String label : labelOrder) {
//					F1Measure res = results.get(label);
//					if (res == null)
//						continue;
//					System.out.println(String.format(
//							"|  %s  |  %6.2f |  %6.2f |  %6.2f |", label,
//							100 * res.getPrecision(), 100 * res.getRecall(),
//							100 * res.getF1()));
//				}
//				System.out.println();
//
//			} catch (Exception e) {
//				LOG.error("Loading testset " + testCorpusFileName, e);
//				System.exit(1);
//			}
//		}

		LOG.info("Saving final model...");
		PrintStream ps;
		try {
			ps = new PrintStream("model.simple");
			hmm.save(ps, inputCorpus.getFeatureEncoding(),
					inputCorpus.getStateEncoding());
			ps.close();
		} catch (FileNotFoundException e) {
			LOG.error("Saving model", e);
		}

		LOG.info("Training done!");
	}

	private static class EvaluateModelListener implements Listener {

		private IobChunkEvaluation eval;

		private SequenceInput[] inputs;
		private SequenceOutput[] outputs;
		private SequenceOutput[] predicteds;

		private static final String[] labelOrder = { "LOC", "MISC", "ORG",
				"PER", "overall" };

		public EvaluateModelListener(SequenceInput[] inputs,
				SequenceOutput[] outputs, StringEncoding stateEncoding,
				String nullLabel) {
			this.inputs = inputs;
			this.outputs = outputs;
			this.predicteds = new SequenceOutput[inputs.length];
			// Allocate output sequences for predictions.
			for (int idx = 0; idx < inputs.length; ++idx)
				predicteds[idx] = (SequenceOutput) inputs[idx].createOutput();
			eval = new IobChunkEvaluation(stateEncoding, nullLabel);
		}

		@Override
		public boolean beforeTraining(Model curModel) {
			return true;
		}

		@Override
		public void afterTraining(Model curModel) {
		}

		@Override
		public boolean beforeEpoch(Model curModel, int epoch, int iteration) {
			return true;
		}

		@Override
		public boolean afterEpoch(Model curModel, int epoch, double loss,
				int iteration) {

			try {
				// Clone the current model to average it.
				curModel = (Model) curModel.clone();
			} catch (CloneNotSupportedException e) {
				LOG.error("Cloning current model on epoch " + epoch
						+ " and iteration " + iteration, e);
				return true;
			}

			// Average the current model.
			curModel.average(iteration);

			// Fill the list of predicted outputs.
			for (int idx = 0; idx < inputs.length; ++idx)
				// Predict (tag the output sequence).
				curModel.inference(inputs[idx], predicteds[idx]);

			// Evaluate the sequences.
			Map<String, F1Measure> results = eval.evaluateSequences(inputs,
					outputs, predicteds);

			// Write results: precision, recall and F-1 values.
			System.out.println();
			System.out.println("Performance after epoch " + epoch);
			System.out.println("|  *Class*  |  *P*  |  *R*  |  *F*  |");
			for (String label : labelOrder) {
				F1Measure res = results.get(label);
				if (res == null)
					continue;
				System.out.println(String.format(
						"|  %s  |  %6.2f |  %6.2f |  %6.2f |", label,
						100 * res.getPrecision(), 100 * res.getRecall(),
						100 * res.getF1()));
			}
			System.out.println();

			return true;
		}

	}
}
