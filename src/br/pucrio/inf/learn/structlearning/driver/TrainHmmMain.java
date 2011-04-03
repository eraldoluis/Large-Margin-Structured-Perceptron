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

import br.pucrio.inf.learn.structlearning.algorithm.AwayFromWorsePerceptron;
import br.pucrio.inf.learn.structlearning.algorithm.LossAugmentedPerceptron;
import br.pucrio.inf.learn.structlearning.algorithm.Perceptron;
import br.pucrio.inf.learn.structlearning.algorithm.Perceptron.Listener;
import br.pucrio.inf.learn.structlearning.algorithm.TowardBetterPerceptron;
import br.pucrio.inf.learn.structlearning.application.sequence.ArrayBasedHmm;
import br.pucrio.inf.learn.structlearning.application.sequence.SequenceInput;
import br.pucrio.inf.learn.structlearning.application.sequence.SequenceOutput;
import br.pucrio.inf.learn.structlearning.application.sequence.ViterbiInference;
import br.pucrio.inf.learn.structlearning.application.sequence.data.Dataset;
import br.pucrio.inf.learn.structlearning.application.sequence.evaluation.F1Measure;
import br.pucrio.inf.learn.structlearning.application.sequence.evaluation.IobChunkEvaluation;
import br.pucrio.inf.learn.structlearning.data.StringEncoding;
import br.pucrio.inf.learn.structlearning.task.Inference;
import br.pucrio.inf.learn.structlearning.task.Model;
import br.pucrio.inf.learn.util.CommandLineOptionsUtil;
import br.pucrio.inf.learn.util.DebugUtil;

public class TrainHmmMain implements Driver.Command {

	private static final Log LOG = LogFactory.getLog(TrainHmmMain.class);

	private static final int NON_ANNOTATED_LABEL_CODE = -33;

	/**
	 * Available training algorithms.
	 */
	private static enum AlgorithmType {
		/**
		 * Ordinary structured Perceptron
		 */
		PERCEPTRON,

		/**
		 * Loss-augmented Perceptron.
		 */
		LOSS_PERCEPTRON,

		/**
		 * Away-from-worse Perceptron (McAllester et al., 2011).
		 */
		AWAY_FROM_WORSE_PERCEPTRON,

		/**
		 * Toward-better Perceptron (McAllester et al., 2011).
		 */
		TOWARD_BETTER_PERCEPTRON
	}

	@SuppressWarnings("static-access")
	@Override
	public void run(String[] args) {
		Options options = new Options();
		options.addOption(OptionBuilder.withLongOpt("incorpus").isRequired()
				.withArgName("input corpus").hasArg().withDescription(
						"Input corpus file name.").create('i'));
		options.addOption(OptionBuilder.withLongOpt("inadd").withArgName(
				"additional corpus[,weight[,step]]").hasArg().withDescription(
				"Additional corpus file name and "
						+ "an optional weight separated by comma and "
						+ "an weight step.").create());
		options.addOption(OptionBuilder.withLongOpt("model").hasArg()
				.withArgName("model filename").withDescription(
						"Name of the file to save the resulting model.")
				.create('o'));
		options.addOption(OptionBuilder.withLongOpt("numepochs").withArgName(
				"number of epochs").hasArg().withDescription(
				"Number of epochs: how many iterations over the"
						+ " training set.").create('T'));
		options.addOption(OptionBuilder.withLongOpt("learnrate").withArgName(
				"learning rate within [0:1]").hasArg().withDescription(
				"Learning rate used in the updates.").create());
		options.addOption(OptionBuilder.withLongOpt("defstate").withArgName(
				"state label").hasArg().withDescription(
				"Default state label to use when all states weight"
						+ " the same.").create('d'));
		options.addOption(OptionBuilder.withLongOpt("nullstate").withArgName(
				"state label").hasArg().withDescription(
				"Null state label if different of default state.").create());
		options.addOption(OptionBuilder.withLongOpt("labels").withArgName(
				"state labels").hasArg().withDescription(
				"List of state labels separated by commas. This can be"
						+ " usefull to specify the preference order of"
						+ " state labels. This option overwrite the"
						+ " following 'tagset' option.").create());
		options.addOption(OptionBuilder.withLongOpt("encoding").withArgName(
				"feature values encoding file").hasArg().withDescription(
				"Filename that contains a list of considered feature"
						+ " values. Any feature value not present in"
						+ " this file is ignored.").create());
		options.addOption(OptionBuilder.withLongOpt("tagset").withArgName(
				"tagset file name").hasArg().withDescription(
				"Name of a file that contains the list of labels, one"
						+ " per line. This can be usefull to specify "
						+ "the preference order of state labels.").create());
		options.addOption(OptionBuilder.withLongOpt("testcorpus").withArgName(
				"test corpus").hasArg().withDescription(
				"Test corpus file name.").create('t'));
		options.addOption(OptionBuilder.withLongOpt("perepoch")
				.withDescription(
						"The evaluation on the test corpus will "
								+ "be performed after each training epoch.")
				.create());
		options.addOption(OptionBuilder.withLongOpt("nonannlabel").withArgName(
				"non-annotated state label").hasArg().withDescription(
				"Set the special state label that indicates "
						+ "non-annotated tokens and, consequently, it "
						+ "will an HMM considering this information").create());
		options.addOption(OptionBuilder.withLongOpt("progress").withArgName(
				"rate of examples").hasArg().withDescription(
				"Rate to report the training progress within each" + " epoch.")
				.create());
		options.addOption(OptionBuilder.withLongOpt("seed").withArgName(
				"integer value").hasArg().withDescription(
				"Random number generator seed.").create());
		options.addOption(OptionBuilder.withLongOpt("lossweight").withArgName(
				"numeric loss weight").hasArg().withDescription(
				"Weight of the loss term in the inference objective"
						+ " function.").create());
		options.addOption(OptionBuilder.withLongOpt("alg").withArgName(
				"training algorithm").hasArg().withDescription(
				"Which training algorithm to be used: "
						+ "perc (ordinary Perceptron), "
						+ "pla (Partial-labeling aware Perceptron), "
						+ "loss (Loss-augmented Perceptron), "
						+ "afworse (away-from-worse Perceptron), "
						+ "tobetter (toward-better Perceptron)").create());
		options.addOption(OptionBuilder.withLongOpt("verbose").withDescription(
				"Print debug information.").create('v'));

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
		String[] inputCorpusFileNames = cmdLine.getOptionValues('i');
		String additionalCorpusFileName = cmdLine.getOptionValue("inadd");
		String modelFileName = cmdLine.getOptionValue('o');
		int numEpochs = Integer.parseInt(cmdLine.getOptionValue('T', "10"));
		double learningRate = Double.parseDouble(cmdLine.getOptionValue(
				"learnrate", "1"));
		String defaultLabel = cmdLine.getOptionValue('d', "0");
		String nullLabel = cmdLine.getOptionValue("nullstate", defaultLabel);
		String testCorpusFileName = cmdLine.getOptionValue('t');
		boolean evalPerEpoch = cmdLine.hasOption("perepoch");
		String labels = cmdLine.getOptionValue("labels");
		String encodingFile = cmdLine.getOptionValue("encoding");
		String tagsetFileName = cmdLine.getOptionValue("tagset");
		String nonAnnotatedLabel = cmdLine.getOptionValue("nonannlabel");
		Double reportProgressRate = Double.parseDouble(cmdLine
				.getOptionValue("progress"));
		String seedStr = cmdLine.getOptionValue("seed");
		double lossWeight = Double.parseDouble(cmdLine.getOptionValue(
				"lossweight", "0d"));
		boolean verbose = cmdLine.hasOption("verbose");

		LOG.info("Loading input corpus...");
		Dataset inputCorpusA = null;
		Dataset inputCorpusB = null;
		double weightAdditionalCorpus = -1d;
		double weightStep = -1d;
		StringEncoding featureEncoding = null;
		StringEncoding stateEncoding = null;
		try {

			// Create (or load) feature values encoding.
			if (encodingFile != null)
				featureEncoding = new StringEncoding(encodingFile);
			else
				featureEncoding = new StringEncoding();

			// Create state labels encoding.
			if (labels != null)
				// State set given in the command-line.
				stateEncoding = new StringEncoding(labels.split(","));
			else if (tagsetFileName != null)
				// State set given in a file.
				stateEncoding = new StringEncoding(tagsetFileName);
			else
				// State set automatically retrieved from training data (codes
				// depend on order of appereance of the labels).
				stateEncoding = new StringEncoding();

			// Get the list of input paths and concatenate the corpora in them.
			inputCorpusA = new Dataset(featureEncoding, stateEncoding,
					nonAnnotatedLabel, NON_ANNOTATED_LABEL_CODE);

			// Load the first data file, which can be the standard input.
			if (inputCorpusFileNames[0].equals("stdin"))
				inputCorpusA.load(System.in);
			else
				inputCorpusA.load(inputCorpusFileNames[0]);

			// Load other data files.
			for (int idxFile = 1; idxFile < inputCorpusFileNames.length; ++idxFile) {
				Dataset other = new Dataset(inputCorpusFileNames[idxFile],
						featureEncoding, stateEncoding, nonAnnotatedLabel,
						NON_ANNOTATED_LABEL_CODE);
				inputCorpusA.add(other);
			}

			if (additionalCorpusFileName != null) {
				if (additionalCorpusFileName.contains(",")) {
					String[] fileNameAndWeight = additionalCorpusFileName
							.split(",");
					additionalCorpusFileName = fileNameAndWeight[0];
					weightAdditionalCorpus = Double
							.parseDouble(fileNameAndWeight[1]);
					if (fileNameAndWeight.length > 2)
						weightStep = Double.parseDouble(fileNameAndWeight[2]);
				}

				inputCorpusB = new Dataset(additionalCorpusFileName,
						featureEncoding, stateEncoding, nonAnnotatedLabel,
						NON_ANNOTATED_LABEL_CODE);
			}

		} catch (Exception e) {
			LOG.error("Loading input corpus", e);
			System.exit(1);
		}

		LOG.info("Allocating initial model...");
		ViterbiInference viterbi = new ViterbiInference(inputCorpusA
				.getStateEncoding().put(defaultLabel));
		ArrayBasedHmm hmm = new ArrayBasedHmm(inputCorpusA.getNumberOfStates(),
				inputCorpusA.getNumberOfSymbols());

		// Parse algorithm type option.
		AlgorithmType algType = null;
		String algTypeStr = cmdLine.getOptionValue("alg");
		if (algTypeStr == null)
			algType = AlgorithmType.PERCEPTRON;
		else if (algTypeStr.equals("perc"))
			algType = AlgorithmType.PERCEPTRON;
		else if (algTypeStr.equals("loss"))
			algType = AlgorithmType.LOSS_PERCEPTRON;
		else if (algTypeStr.equals("afworse"))
			algType = AlgorithmType.AWAY_FROM_WORSE_PERCEPTRON;
		else if (algTypeStr.equals("tobetter"))
			algType = AlgorithmType.TOWARD_BETTER_PERCEPTRON;
		else {
			System.err.println("Unknown algorithm: " + algTypeStr);
			CommandLineOptionsUtil.usage(getClass().getSimpleName(), options);
			System.exit(1);
		}

		// Create the chosen algorithm.
		Perceptron alg = null;
		switch (algType) {
		case PERCEPTRON:
			// Ordinary Perceptron implementation (Collins'): does not consider
			// neither partially-annotated examples nor customized loss
			// functions.
			alg = new Perceptron(viterbi, hmm, numEpochs, learningRate);
			break;
		case LOSS_PERCEPTRON:
			// Loss-augumented implementation: considers partially-labeled
			// examples and customized loss function (per-token
			// misclassification loss).
			alg = new LossAugmentedPerceptron(viterbi, hmm, numEpochs,
					learningRate, lossWeight);
			break;
		case AWAY_FROM_WORSE_PERCEPTRON:
			// Away-from-worse implementation.
			alg = new AwayFromWorsePerceptron(viterbi, hmm, numEpochs,
					learningRate, lossWeight);
			break;
		case TOWARD_BETTER_PERCEPTRON:
			// Toward-better implementation.
			alg = new TowardBetterPerceptron(viterbi, hmm, numEpochs,
					learningRate, lossWeight);
			break;
		}

		if (nonAnnotatedLabel != null) {
			// Non-annotated state label was specified and therefore the input
			// dataset can contain non-annotated tokens that must be properly
			// tackled by the inference algorithm.
			viterbi.setNonAnnotatedStateCode(NON_ANNOTATED_LABEL_CODE);
			// Signal the presence of partially-labeled examples to the
			// algorithm.
			alg.setPartiallyAnnotatedExamples(true);
		}

		if (seedStr != null)
			// User provided seed to random number generator.
			alg.setSeed(Long.parseLong(seedStr));

		if (reportProgressRate != null)
			// Progress report rate.
			alg.setReportProgressRate(reportProgressRate);

		// Ignore features not seen in the training corpus.
		inputCorpusA.getFeatureEncoding().setReadOnly(true);
		inputCorpusA.getStateEncoding().setReadOnly(true);

		// Evaluation after each training epoch.
		if (testCorpusFileName != null && evalPerEpoch) {
			try {

				LOG.info("Loading and preparing test data...");
				Dataset testset = new Dataset(testCorpusFileName, inputCorpusA
						.getFeatureEncoding(), inputCorpusA.getStateEncoding());
				alg.setListener(new EvaluateModelListener(testset.getInputs(),
						testset.getOutputs(), inputCorpusA.getStateEncoding(),
						nullLabel));

			} catch (Exception e) {
				LOG.error("Loading testset " + testCorpusFileName, e);
				System.exit(1);
			}
		}

		if (verbose) {
			DebugUtil.featureEncoding = featureEncoding;
			DebugUtil.stateEncoding = stateEncoding;
			DebugUtil.print = true;
		}

		LOG.info("Training model...");
		if (inputCorpusB == null) {
			// Train on only one dataset.
			alg.train(inputCorpusA.getInputs(), inputCorpusA.getOutputs(),
					inputCorpusA.getFeatureEncoding(), inputCorpusA
							.getStateEncoding());
		} else {
			// Train on two datasets.
			if (weightAdditionalCorpus < 0d)
				// If no different weight was given for the B dataset, then use
				// a weight proportional to the sizes of the datasets.
				weightAdditionalCorpus = ((double) inputCorpusB
						.getNumberOfExamples())
						/ (inputCorpusA.getNumberOfExamples() + inputCorpusB
								.getNumberOfExamples());
			alg.train(inputCorpusA.getInputs(), inputCorpusA.getOutputs(),
					1d - weightAdditionalCorpus, weightStep, inputCorpusB
							.getInputs(), inputCorpusB.getOutputs(),
					inputCorpusA.getFeatureEncoding(), inputCorpusA
							.getStateEncoding());
		}

		// Evaluation only for the final model.
		if (testCorpusFileName != null && !evalPerEpoch) {
			try {

				LOG.info("Loading and preparing test data...");
				Dataset testset = new Dataset(testCorpusFileName, inputCorpusA
						.getFeatureEncoding(), inputCorpusA.getStateEncoding());

				// Allocate output sequences for predictions.
				SequenceInput[] inputs = testset.getInputs();
				SequenceOutput[] outputs = testset.getOutputs();
				SequenceOutput[] predicteds = new SequenceOutput[inputs.length];
				for (int idx = 0; idx < inputs.length; ++idx)
					predicteds[idx] = (SequenceOutput) inputs[idx]
							.createOutput();
				IobChunkEvaluation eval = new IobChunkEvaluation(inputCorpusA
						.getStateEncoding(), nullLabel);

				// Fill the list of predicted outputs.
				for (int idx = 0; idx < inputs.length; ++idx)
					// Predict (tag the output sequence).
					viterbi.inference(hmm, inputs[idx], predicteds[idx]);

				// Evaluate the sequences.
				Map<String, F1Measure> results = eval.evaluateSequences(inputs,
						outputs, predicteds);

				// Write results: precision, recall and F-1 values.
				System.out.println();
				System.out.println("|  *Class*  |  *P*  |  *R*  |  *F*  |");
				String[] labelOrder = { "LOC", "MISC", "ORG", "PER", "overall" };
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

			} catch (Exception e) {
				LOG.error("Loading testset " + testCorpusFileName, e);
				System.exit(1);
			}
		}

		if (modelFileName != null) {
			LOG.info("Saving final model...");
			PrintStream ps;
			try {
				ps = new PrintStream(modelFileName);
				hmm.save(ps, inputCorpusA.getFeatureEncoding(), inputCorpusA
						.getStateEncoding());
				ps.close();
			} catch (FileNotFoundException e) {
				LOG.error("Saving model " + modelFileName, e);
			}
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
		public boolean afterEpoch(Inference viterbi, Model hmm, int epoch,
				double loss, int iteration) {

			try {
				// Clone the current model to average it.
				hmm = (Model) hmm.clone();
			} catch (CloneNotSupportedException e) {
				LOG.error("Cloning current model on epoch " + epoch
						+ " and iteration " + iteration, e);
				return true;
			}

			// Average the current model.
			hmm.average(iteration);

			// Fill the list of predicted outputs.
			for (int idx = 0; idx < inputs.length; ++idx)
				// Predict (tag the output sequence).
				viterbi.inference(hmm, inputs[idx], predicteds[idx]);

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
						"|  %s  |  %6.2f |  %6.2f |  %6.2f |", label, 100 * res
								.getPrecision(), 100 * res.getRecall(),
						100 * res.getF1()));
			}
			System.out.println();

			return true;
		}

	}
}
