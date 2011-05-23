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

import br.pucrio.inf.learn.structlearning.algorithm.TrainingListener;
import br.pucrio.inf.learn.structlearning.algorithm.perceptron.AwayFromWorsePerceptron;
import br.pucrio.inf.learn.structlearning.algorithm.perceptron.LossAugmentedPerceptron;
import br.pucrio.inf.learn.structlearning.algorithm.perceptron.Perceptron;
import br.pucrio.inf.learn.structlearning.algorithm.perceptron.Perceptron.LearnRateUpdateStrategy;
import br.pucrio.inf.learn.structlearning.algorithm.perceptron.TowardBetterPerceptron;
import br.pucrio.inf.learn.structlearning.application.sequence.AveragedArrayBasedHmm;
import br.pucrio.inf.learn.structlearning.application.sequence.SequenceInput;
import br.pucrio.inf.learn.structlearning.application.sequence.SequenceOutput;
import br.pucrio.inf.learn.structlearning.application.sequence.ViterbiInference;
import br.pucrio.inf.learn.structlearning.application.sequence.data.Dataset;
import br.pucrio.inf.learn.structlearning.application.sequence.evaluation.F1Measure;
import br.pucrio.inf.learn.structlearning.application.sequence.evaluation.IobChunkEvaluation;
import br.pucrio.inf.learn.structlearning.data.FeatureEncoding;
import br.pucrio.inf.learn.structlearning.data.JavaHashCodeEncoding;
import br.pucrio.inf.learn.structlearning.data.Lookup3Encoding;
import br.pucrio.inf.learn.structlearning.data.Murmur2Encoding;
import br.pucrio.inf.learn.structlearning.data.Murmur3Encoding;
import br.pucrio.inf.learn.structlearning.data.StringMapEncoding;
import br.pucrio.inf.learn.structlearning.driver.Driver.Command;
import br.pucrio.inf.learn.structlearning.task.Inference;
import br.pucrio.inf.learn.structlearning.task.Model;
import br.pucrio.inf.learn.util.CommandLineOptionsUtil;
import br.pucrio.inf.learn.util.DebugUtil;

public class TrainHmmMain implements Command {

	private static final Log LOG = LogFactory.getLog(TrainHmmMain.class);

	private static final int NON_ANNOTATED_LABEL_CODE = -10;

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
		options.addOption(OptionBuilder
				.withLongOpt("alg")
				.withArgName("perc | loss | afworse | tobetter")
				.hasArg()
				.withDescription(
						"Which training algorithm to be used: "
								+ "perc (ordinary Perceptron), "
								+ "loss (Loss-augmented Perceptron), "
								+ "afworse (away-from-worse Perceptron), "
								+ "tobetter (toward-better Perceptron)")
				.create());
		options.addOption(OptionBuilder.withLongOpt("incorpus").isRequired()
				.withArgName("input corpus").hasArg()
				.withDescription("Input corpus file name.").create('i'));
		options.addOption(OptionBuilder
				.withLongOpt("inadd")
				.withArgName("additional corpus[,weight[,step]]")
				.hasArg()
				.withDescription(
						"Additional corpus file name and "
								+ "an optional weight separated by comma and "
								+ "an weight step.").create());
		options.addOption(OptionBuilder
				.withLongOpt("model")
				.hasArg()
				.withArgName("model filename")
				.withDescription(
						"Name of the file to save the resulting model.")
				.create('o'));
		options.addOption(OptionBuilder
				.withLongOpt("numepochs")
				.withArgName("number of epochs")
				.hasArg()
				.withDescription(
						"Number of epochs: how many iterations over the"
								+ " training set.").create('T'));
		options.addOption(OptionBuilder.withLongOpt("learnrate")
				.withArgName("learning rate within [0:1]").hasArg()
				.withDescription("Learning rate used in the updates.").create());
		options.addOption(OptionBuilder
				.withLongOpt("defstate")
				.withArgName("state label")
				.hasArg()
				.withDescription(
						"Default state label to use when all states weight"
								+ " the same.").create('d'));
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
						"List of state labels separated by commas. This can be"
								+ " usefull to specify the preference order of"
								+ " state labels. This option overwrite the"
								+ " following 'tagset' option.").create());
		options.addOption(OptionBuilder
				.withLongOpt("encoding")
				.withArgName("feature values encoding file")
				.hasArg()
				.withDescription(
						"Filename that contains a list of considered feature"
								+ " values. Any feature value not present in"
								+ " this file is ignored.").create());
		options.addOption(OptionBuilder
				.withLongOpt("murmur")
				.withArgName("size,seed")
				.hasArg()
				.withDescription(
						"Use a Murmur3 hash function to encode the feature values. "
								+ "This option can be very memory-efficient and "
								+ "handful since the amount of memory needed to"
								+ " store the model is linearly proportional to"
								+ " this number. If this number has the suffix "
								+ "'b' then it is considered as the number of"
								+ " bits needed to encode a feature code, i.e.,"
								+ " the proper size of the hash table will be"
								+ " 2^n, where n is the specified number of"
								+ " bits.").create());
		options.addOption(OptionBuilder
				.withLongOpt("murmur3")
				.withArgName("size,seed")
				.hasArg()
				.withDescription(
						"Use a Murmur3 hash function to encode the feature values. "
								+ "This option can be very memory-efficient and "
								+ "handful since the amount of memory needed to"
								+ " store the model is linearly proportional to"
								+ " this number. If this number has the suffix "
								+ "'b' then it is considered as the number of"
								+ " bits needed to encode a feature code, i.e.,"
								+ " the proper size of the hash table will be"
								+ " 2^n, where n is the specified number of"
								+ " bits.").create());
		options.addOption(OptionBuilder
				.withLongOpt("murmur2")
				.withArgName("size,seed")
				.hasArg()
				.withDescription(
						"Use a Murmur2 hash function to encode the feature values. "
								+ "This option can be very memory-efficient and "
								+ "handful since the amount of memory needed to"
								+ " store the model is linearly proportional to"
								+ " this number. If this number has the suffix "
								+ "'b' then it is considered as the number of"
								+ " bits needed to encode a feature code, i.e.,"
								+ " the proper size of the hash table will be"
								+ " 2^n, where n is the specified number of"
								+ " bits.").create());
		options.addOption(OptionBuilder
				.withLongOpt("lookup3")
				.withArgName("size,seed")
				.hasArg()
				.withDescription(
						"Use a Lookup3 hash function to encode the feature values. "
								+ "This option can be very memory-efficient and "
								+ "handful since the amount of memory needed to"
								+ " store the model is linearly proportional to"
								+ " this number. If this number has the suffix "
								+ "'b' then it is considered as the number of"
								+ " bits needed to encode a feature code, i.e.,"
								+ " the proper size of the hash table will be"
								+ " 2^n, where n is the specified number of"
								+ " bits.").create());
		options.addOption(OptionBuilder
				.withLongOpt("javahash")
				.withArgName("hash table size")
				.hasArg()
				.withDescription(
						"Use the default Java hashing function (hashCode method) "
								+ "to encode feature values.").create());
		options.addOption(OptionBuilder
				.withLongOpt("tagset")
				.withArgName("tagset file name")
				.hasArg()
				.withDescription(
						"Name of a file that contains the list of labels, one"
								+ " per line. This can be usefull to specify "
								+ "the preference order of state labels.")
				.create());
		options.addOption(OptionBuilder.withLongOpt("testcorpus")
				.withArgName("test corpus").hasArg()
				.withDescription("Test corpus file name.").create('t'));
		options.addOption(OptionBuilder
				.withLongOpt("perepoch")
				.withDescription(
						"The evaluation on the test corpus will "
								+ "be performed after each training epoch.")
				.create());
		options.addOption(OptionBuilder
				.withLongOpt("nonannlabel")
				.withArgName("non-annotated state label")
				.hasArg()
				.withDescription(
						"Set the special state label that indicates "
								+ "non-annotated tokens and, consequently, it "
								+ "will an HMM considering this information")
				.create());
		options.addOption(OptionBuilder
				.withLongOpt("progress")
				.withArgName("rate of examples")
				.hasArg()
				.withDescription(
						"Rate to report the training progress within each"
								+ " epoch.").create());
		options.addOption(OptionBuilder.withLongOpt("seed")
				.withArgName("integer value").hasArg()
				.withDescription("Random number generator seed.").create());
		options.addOption(OptionBuilder
				.withLongOpt("lossweight")
				.withArgName("numeric loss weight")
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
		options.addOption(OptionBuilder
				.withLongOpt("lrupdate")
				.withArgName("none | linear | quadratic | root")
				.hasArg()
				.withDescription(
						"Which learning rate update strategy to be used. Valid "
								+ "values are: "
								+ "none (constant learning rate), "
								+ "linear (n/t), "
								+ "quadratic (n/(t*t)) or "
								+ "root (n/sqrt(t)), "
								+ "where n is the initial learning rate and t "
								+ "is the current iteration (number of processed"
								+ " examples).").create());
		options.addOption(OptionBuilder
				.withLongOpt("lossnonlabeledweight")
				.withArgName("numeric weight")
				.hasArg()
				.withDescription(
						"Specify a different loss weight for non-annotated tokens.")
				.create());
		options.addOption(OptionBuilder
				.withLongOpt("lossnonlabeledweightinc")
				.withArgName("numeric increment per epoch")
				.hasArg()
				.withDescription(
						"Specify an increment (per epoch) to the loss weight on non-annotated tokens."
								+ " The maximum value for this weight is the annotated tokens loss weight.")
				.create());
		options.addOption(OptionBuilder.withLongOpt("debug")
				.withDescription("Print debug information.").create());
		options.addOption(OptionBuilder
				.withLongOpt("skipunlabeled")
				.withDescription(
						"Skip completely unlabeled sequences in the input corpora.")
				.create());
		options.addOption(OptionBuilder
				.withLongOpt("norm")
				.withDescription(
						"Normalize the input structures before "
								+ "training and testing.").create());

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
		String[] inputCorpusFileNames = cmdLine.getOptionValues("incorpus");
		String additionalCorpusFileName = cmdLine.getOptionValue("inadd");
		String modelFileName = cmdLine.getOptionValue("model");
		int numEpochs = Integer.parseInt(cmdLine.getOptionValue("numepochs",
				"10"));
		double learningRate = Double.parseDouble(cmdLine.getOptionValue(
				"learnrate", "1"));
		String defaultLabel = cmdLine.getOptionValue("defstate", "0");
		String nullLabel = cmdLine.getOptionValue("nullstate", defaultLabel);
		String testCorpusFileName = cmdLine.getOptionValue("testCorpus");
		boolean evalPerEpoch = cmdLine.hasOption("perepoch");
		String labels = cmdLine.getOptionValue("labels");
		String encodingFile = cmdLine.getOptionValue("encoding");
		String murmur = cmdLine.getOptionValue("murmur");
		String murmur3 = cmdLine.getOptionValue("murmur3");
		String murmur2 = cmdLine.getOptionValue("murmur2");
		String lookup3 = cmdLine.getOptionValue("lookup3");
		String javaHashSizeStr = cmdLine.getOptionValue("javahash");
		String tagsetFileName = cmdLine.getOptionValue("tagset");
		String nonAnnotatedLabel = cmdLine.getOptionValue("nonannlabel");
		Double reportProgressRate = Double.parseDouble(cmdLine
				.getOptionValue("progress"));
		String seedStr = cmdLine.getOptionValue("seed");
		double lossWeight = Double.parseDouble(cmdLine.getOptionValue(
				"lossweight", "0d"));
		boolean averageWeights = !cmdLine.hasOption("noavg");
		String lrUpdateStrategy = cmdLine.getOptionValue("lrupdate");
		String lossNonAnnotatedWeightStr = cmdLine
				.getOptionValue("lossnonlabeledweight");
		double lossNonAnnotatedWeightInc = Double.parseDouble(cmdLine
				.getOptionValue("lossnonlabeledweightinc", "0d"));
		boolean debug = cmdLine.hasOption("debug");
		boolean skipCompletelyNonAnnotatedExamples = cmdLine
				.hasOption("skipunlabeled");
		boolean normalizeInput = cmdLine.hasOption("norm");

		LOG.info("Loading input corpus...");
		Dataset inputCorpusA = null;
		Dataset inputCorpusB = null;
		double weightAdditionalCorpus = -1d;
		double weightStep = -1d;
		FeatureEncoding<String> featureEncoding = null;
		StringMapEncoding stateEncoding = null;
		try {

			// Create (or load) the feature value encoding.
			if (encodingFile != null) {

				// Load a map-based encoding from the given file. Thus, the
				// feature values present in this file will be encoded
				// unambiguously but any unknown value will be ignored.
				featureEncoding = new StringMapEncoding(encodingFile);

			} else if (murmur != null) {

				// Create a feature encoding based on the Murmur3 hash function.
				int size = parseValueDirectOrBits(murmur);
				int seed = parseEncodingSeed(murmur);
				if (seed != Integer.MIN_VALUE)
					featureEncoding = new Murmur3Encoding(size);
				else
					featureEncoding = new Murmur3Encoding(size, seed);

			} else if (murmur3 != null) {

				// Create a feature encoding based on the Murmur3 hash function.
				int size = parseValueDirectOrBits(murmur3);
				int seed = parseEncodingSeed(murmur3);
				if (seed != Integer.MIN_VALUE)
					featureEncoding = new Murmur3Encoding(size);
				else
					featureEncoding = new Murmur3Encoding(size, seed);

			} else if (murmur2 != null) {

				// Create a feature encoding based on the Murmur2 hash function.
				int size = parseValueDirectOrBits(murmur2);
				int seed = parseEncodingSeed(murmur2);
				if (seed != Integer.MIN_VALUE)
					featureEncoding = new Murmur2Encoding(size);
				else
					featureEncoding = new Murmur2Encoding(size, seed);

			} else if (lookup3 != null) {

				// Create a feature encoding based on the Lookup3 hash function.
				int size = parseValueDirectOrBits(lookup3);
				int seed = parseEncodingSeed(lookup3);
				if (seed != Integer.MIN_VALUE)
					featureEncoding = new Lookup3Encoding(size);
				else
					featureEncoding = new Lookup3Encoding(size, seed);

			} else if (javaHashSizeStr != null) {

				// Create a feature encoding based on the Java hash function.
				featureEncoding = new JavaHashCodeEncoding(
						parseValueDirectOrBits(javaHashSizeStr));

			} else {

				// Create an empty and flexible feature encoding that will
				// encode unambiguously all feature values.
				featureEncoding = new StringMapEncoding();

			}

			// Create or load the state label encoding.
			if (labels != null)
				// State set given in the command-line.
				stateEncoding = new StringMapEncoding(labels.split(","));
			else if (tagsetFileName != null)
				// State set given in a file.
				stateEncoding = new StringMapEncoding(tagsetFileName);
			else
				// State set automatically retrieved from training data (codes
				// depend on order of appereance of the labels).
				stateEncoding = new StringMapEncoding();

			// Get the list of input paths and concatenate the corpora in them.
			inputCorpusA = new Dataset(featureEncoding, stateEncoding,
					nonAnnotatedLabel, NON_ANNOTATED_LABEL_CODE);
			inputCorpusA
					.setSkipCompletelyNonAnnotatedExamples(skipCompletelyNonAnnotatedExamples);

			// Load the first data file, which can be the standard input.
			if (inputCorpusFileNames[0].equals("stdin"))
				inputCorpusA.load(System.in);
			else
				inputCorpusA.load(inputCorpusFileNames[0]);

			// Load other data files.
			for (int idxFile = 1; idxFile < inputCorpusFileNames.length; ++idxFile) {
				Dataset other = new Dataset(inputCorpusFileNames[idxFile],
						featureEncoding, stateEncoding, nonAnnotatedLabel,
						NON_ANNOTATED_LABEL_CODE,
						skipCompletelyNonAnnotatedExamples);
				inputCorpusA.add(other);
			}

			if (normalizeInput) {
				LOG.info("Normalizing input structures...");
				// Normalize the input structures.
				inputCorpusA.normalizeInputStructures(inputCorpusA
						.getMaxNumberOfEmissionFeatures());
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
						NON_ANNOTATED_LABEL_CODE,
						skipCompletelyNonAnnotatedExamples);

				if (normalizeInput)
					// Normalize the input structures.
					inputCorpusB.normalizeInputStructures(inputCorpusB
							.getMaxNumberOfEmissionFeatures());
			}

		} catch (Exception e) {
			LOG.error("Parsing command-line options", e);
			System.exit(1);
		}

		LOG.info("Allocating initial model...");
		ViterbiInference viterbiInference = new ViterbiInference(inputCorpusA
				.getStateEncoding().put(defaultLabel));
		AveragedArrayBasedHmm hmm = new AveragedArrayBasedHmm(
				inputCorpusA.getNumberOfStates(),
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

		// Learning rate update strategy.
		LearnRateUpdateStrategy learningRateUpdateStrategy = LearnRateUpdateStrategy.NONE;
		if (lrUpdateStrategy == null)
			learningRateUpdateStrategy = LearnRateUpdateStrategy.NONE;
		else if (lrUpdateStrategy.equals("none"))
			learningRateUpdateStrategy = LearnRateUpdateStrategy.NONE;
		else if (lrUpdateStrategy.equals("linear"))
			learningRateUpdateStrategy = LearnRateUpdateStrategy.LINEAR;
		else if (lrUpdateStrategy.equals("quadratic"))
			learningRateUpdateStrategy = LearnRateUpdateStrategy.QUADRATIC;
		else if (lrUpdateStrategy.equals("root"))
			learningRateUpdateStrategy = LearnRateUpdateStrategy.SQUARE_ROOT;
		else {
			System.err.println("Unknown learning rate update strategy: "
					+ lrUpdateStrategy);
			CommandLineOptionsUtil.usage(getClass().getSimpleName(), options);
			System.exit(1);
		}

		// Create the chosen algorithm.
		Perceptron alg = null;
		switch (algType) {
		case PERCEPTRON:
			// Ordinary Perceptron implementation (Collins'): does not consider
			// customized loss functions.
			alg = new Perceptron(viterbiInference, hmm, numEpochs,
					learningRate, true, averageWeights,
					learningRateUpdateStrategy);
			break;
		case LOSS_PERCEPTRON:
			// Loss-augumented implementation: considers customized loss
			// function (per-token misclassification loss).
			if (lossNonAnnotatedWeightStr == null)
				alg = new LossAugmentedPerceptron(viterbiInference, hmm,
						numEpochs, learningRate, lossWeight, true,
						averageWeights, learningRateUpdateStrategy);
			else
				alg = new LossAugmentedPerceptron(viterbiInference, hmm,
						numEpochs, learningRate, lossWeight,
						Double.parseDouble(lossNonAnnotatedWeightStr),
						lossNonAnnotatedWeightInc, true, averageWeights,
						learningRateUpdateStrategy);
			break;
		case AWAY_FROM_WORSE_PERCEPTRON:
			// Away-from-worse implementation.
			if (lossNonAnnotatedWeightStr == null)
				alg = new AwayFromWorsePerceptron(viterbiInference, hmm,
						numEpochs, learningRate, lossWeight, true,
						averageWeights, learningRateUpdateStrategy);
			else
				alg = new AwayFromWorsePerceptron(viterbiInference, hmm,
						numEpochs, learningRate, lossWeight,
						Double.parseDouble(lossNonAnnotatedWeightStr),
						lossNonAnnotatedWeightInc, true, averageWeights,
						learningRateUpdateStrategy);
			break;
		case TOWARD_BETTER_PERCEPTRON:
			// Toward-better implementation.
			if (lossNonAnnotatedWeightStr == null)
				alg = new TowardBetterPerceptron(viterbiInference, hmm,
						numEpochs, learningRate, lossWeight, true,
						averageWeights, learningRateUpdateStrategy);
			else
				alg = new TowardBetterPerceptron(viterbiInference, hmm,
						numEpochs, learningRate, lossWeight,
						Double.parseDouble(lossNonAnnotatedWeightStr),
						lossNonAnnotatedWeightInc, true, averageWeights,
						learningRateUpdateStrategy);
			break;
		}

		if (nonAnnotatedLabel != null) {
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
				Dataset testset = new Dataset(testCorpusFileName,
						inputCorpusA.getFeatureEncoding(),
						inputCorpusA.getStateEncoding());

				if (normalizeInput)
					// Normalize the input structures.
					testset.normalizeInputStructures(testset
							.getMaxNumberOfEmissionFeatures());

				alg.setListener(new EvaluateModelListener(testset.getInputs(),
						testset.getOutputs(), inputCorpusA.getStateEncoding(),
						nullLabel, averageWeights));

			} catch (Exception e) {
				LOG.error("Loading testset " + testCorpusFileName, e);
				System.exit(1);
			}
		}

		// Debug information.
		if (debug) {
			DebugUtil.featureEncoding = featureEncoding;
			DebugUtil.stateEncoding = stateEncoding;
			DebugUtil.print = true;
		}

		LOG.info("Training model...");
		if (inputCorpusB == null) {
			// Train on only one dataset.
			alg.train(inputCorpusA.getInputs(), inputCorpusA.getOutputs(),
					inputCorpusA.getFeatureEncoding(),
					inputCorpusA.getStateEncoding());
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
					1d - weightAdditionalCorpus, weightStep,
					inputCorpusB.getInputs(), inputCorpusB.getOutputs(),
					inputCorpusA.getFeatureEncoding(),
					inputCorpusA.getStateEncoding());
		}

		// Evaluation only for the final model.
		if (testCorpusFileName != null && !evalPerEpoch) {
			try {

				LOG.info("Loading and preparing test data...");
				Dataset testset = new Dataset(testCorpusFileName,
						inputCorpusA.getFeatureEncoding(),
						inputCorpusA.getStateEncoding());

				if (normalizeInput)
					// Normalize the input structures.
					testset.normalizeInputStructures(testset
							.getMaxNumberOfEmissionFeatures());

				// Allocate output sequences for predictions.
				SequenceInput[] inputs = testset.getInputs();
				SequenceOutput[] outputs = testset.getOutputs();
				SequenceOutput[] predicteds = new SequenceOutput[inputs.length];
				for (int idx = 0; idx < inputs.length; ++idx)
					predicteds[idx] = (SequenceOutput) inputs[idx]
							.createOutput();
				IobChunkEvaluation eval = new IobChunkEvaluation(
						inputCorpusA.getStateEncoding(), nullLabel);

				// Fill the list of predicted outputs.
				for (int idx = 0; idx < inputs.length; ++idx)
					// Predict (tag the output sequence).
					viterbiInference.inference(hmm, inputs[idx],
							predicteds[idx]);

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
				hmm.save(ps, inputCorpusA.getFeatureEncoding(),
						inputCorpusA.getStateEncoding());
				ps.close();
			} catch (FileNotFoundException e) {
				LOG.error("Saving model " + modelFileName, e);
			}
		}

		LOG.info("Training done!");
	}

	/**
	 * Convert a string that can specify a value directly or in bits. If the
	 * string ends with a b, the value is considered to be n=log_2(v), where v
	 * is the specified value and n is the returned value (the value of
	 * interest).
	 * 
	 * @param valStr
	 * @return
	 */
	private static int parseValueDirectOrBits(String valStr) {
		if (valStr.contains(","))
			valStr = valStr.split("[,]")[0];

		if (valStr.endsWith("b")) {
			// The size is specified in bits.
			String bitsStr = valStr.substring(0, valStr.length() - 1);
			int bits = Integer.parseInt(bitsStr);
			return ((int) Math.round(Math.pow(2, bits))) - 1;
		}

		// The size is specified directly.
		return Integer.parseInt(valStr);
	}

	/**
	 * Parse the string given as parameter in the encoding option and return the
	 * specified seed value or -Integer.MIN_VALUE otherwise.
	 * 
	 * @param valStr
	 * @return
	 */
	private static int parseEncodingSeed(String valStr) {
		if (valStr.contains(","))
			return Integer.parseInt(valStr.split("[,]")[1]);
		return Integer.MIN_VALUE;
	}

	/**
	 * Training listener to evaluate models after each iteration.
	 * 
	 * @author eraldof
	 * 
	 */
	private static class EvaluateModelListener implements TrainingListener {

		private IobChunkEvaluation eval;

		private SequenceInput[] inputs;

		private SequenceOutput[] outputs;

		private SequenceOutput[] predicteds;

		private boolean averageWeights;

		private static final String[] labelOrder = { "LOC", "MISC", "ORG",
				"PER", "overall" };

		public EvaluateModelListener(SequenceInput[] inputs,
				SequenceOutput[] outputs,
				FeatureEncoding<String> stateEncoding, String nullLabel,
				boolean averageWeights) {
			this.inputs = inputs;
			this.outputs = outputs;
			this.predicteds = new SequenceOutput[inputs.length];
			// Allocate output sequences for predictions.
			for (int idx = 0; idx < inputs.length; ++idx)
				predicteds[idx] = (SequenceOutput) inputs[idx].createOutput();
			this.eval = new IobChunkEvaluation(stateEncoding, nullLabel);
			this.averageWeights = averageWeights;
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
		public boolean afterEpoch(Inference inferenceImpl, Model hmm,
				int epoch, double loss, int iteration) {

			if (averageWeights) {
				try {
					// Clone the current model to average it, if necessary.
					hmm = (Model) hmm.clone();
				} catch (CloneNotSupportedException e) {
					LOG.error("Cloning current model on epoch " + epoch
							+ " and iteration " + iteration, e);
					return true;
				}
			}

			// Average the current model.
			if (averageWeights)
				hmm.average(iteration);

			// Fill the list of predicted outputs.
			for (int idx = 0; idx < inputs.length; ++idx)
				// Predict (tag the output sequence).
				inferenceImpl.inference(hmm, inputs[idx], predicteds[idx]);

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
