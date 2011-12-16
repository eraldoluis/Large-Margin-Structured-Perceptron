package br.pucrio.inf.learn.structlearning.discriminative.driver;

import java.io.FileNotFoundException;
import java.io.PrintStream;
import java.util.Map;
import java.util.Map.Entry;

import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.OptionBuilder;
import org.apache.commons.cli.Options;
import org.apache.commons.cli.ParseException;
import org.apache.commons.cli.PosixParser;
import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;

import br.pucrio.inf.learn.structlearning.discriminative.algorithm.OnlineStructuredAlgorithm.LearnRateUpdateStrategy;
import br.pucrio.inf.learn.structlearning.discriminative.algorithm.TrainingListener;
import br.pucrio.inf.learn.structlearning.discriminative.algorithm.perceptron.AwayFromWorsePerceptron;
import br.pucrio.inf.learn.structlearning.discriminative.algorithm.perceptron.DualLossAugmentedPerceptron;
import br.pucrio.inf.learn.structlearning.discriminative.algorithm.perceptron.LossAugmentedPerceptron;
import br.pucrio.inf.learn.structlearning.discriminative.algorithm.perceptron.Perceptron;
import br.pucrio.inf.learn.structlearning.discriminative.algorithm.perceptron.TowardBetterPerceptron;
import br.pucrio.inf.learn.structlearning.discriminative.application.sequence.AveragedArrayHmm;
import br.pucrio.inf.learn.structlearning.discriminative.application.sequence.AveragedArrayHmm2ndOrder;
import br.pucrio.inf.learn.structlearning.discriminative.application.sequence.AveragedMapHmm;
import br.pucrio.inf.learn.structlearning.discriminative.application.sequence.DualHmm;
import br.pucrio.inf.learn.structlearning.discriminative.application.sequence.Viterbi2ndOrderInference;
import br.pucrio.inf.learn.structlearning.discriminative.application.sequence.ViterbiInference;
import br.pucrio.inf.learn.structlearning.discriminative.application.sequence.data.SequenceDataset;
import br.pucrio.inf.learn.structlearning.discriminative.application.sequence.data.SequenceInput;
import br.pucrio.inf.learn.structlearning.discriminative.application.sequence.data.SequenceOutput;
import br.pucrio.inf.learn.structlearning.discriminative.application.sequence.evaluation.IobChunkEvaluation;
import br.pucrio.inf.learn.structlearning.discriminative.application.sequence.evaluation.LabeledTokenEvaluation;
import br.pucrio.inf.learn.structlearning.discriminative.data.encoding.FeatureEncoding;
import br.pucrio.inf.learn.structlearning.discriminative.data.encoding.HybridStringEncoding;
import br.pucrio.inf.learn.structlearning.discriminative.data.encoding.JavaHashCodeEncoding;
import br.pucrio.inf.learn.structlearning.discriminative.data.encoding.Lookup3Encoding;
import br.pucrio.inf.learn.structlearning.discriminative.data.encoding.Murmur2Encoding;
import br.pucrio.inf.learn.structlearning.discriminative.data.encoding.Murmur3Encoding;
import br.pucrio.inf.learn.structlearning.discriminative.data.encoding.StringMapEncoding;
import br.pucrio.inf.learn.structlearning.discriminative.driver.Driver.Command;
import br.pucrio.inf.learn.structlearning.discriminative.evaluation.EntityF1Evaluation;
import br.pucrio.inf.learn.structlearning.discriminative.evaluation.F1Measure;
import br.pucrio.inf.learn.structlearning.discriminative.task.DualModel;
import br.pucrio.inf.learn.structlearning.discriminative.task.Inference;
import br.pucrio.inf.learn.structlearning.discriminative.task.Model;
import br.pucrio.inf.learn.util.CommandLineOptionsUtil;
import br.pucrio.inf.learn.util.DebugUtil;

/**
 * Driver to discriminatively train a sequential model (HMM) using
 * perceptron-based algorithms.
 * 
 * @author eraldo
 * 
 */
public class TrainHmm implements Command {

	/**
	 * Logging object.
	 */
	private static final Log LOG = LogFactory.getLog(TrainHmm.class);

	/**
	 * Type of task to be performed.
	 */
	public static enum TaskType {
		/**
		 * IOB sequence labeling task.
		 */
		IOB,

		/**
		 * Token labeling task.
		 */
		TOKEN
	}

	/**
	 * Available training algorithms.
	 */
	public static enum AlgorithmType {
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
		TOWARD_BETTER_PERCEPTRON,

		/**
		 * Dual (kernelized) loss-augmented perceptron.
		 */
		DUAL_PERCEPTRON,
	}

	@SuppressWarnings("static-access")
	@Override
	public void run(String[] args) {
		Options options = new Options();
		options.addOption(OptionBuilder
				.withLongOpt("task")
				.withArgName("iob | token")
				.hasArg()
				.withDescription(
						"Which type of task is performed: IOB sequence "
								+ "labeling or token labeling").create());
		options.addOption(OptionBuilder
				.withLongOpt("structure")
				.withArgName("hmm | hmm2")
				.hasArg()
				.withDescription(
						"Which structure to use: hmm (first-order HMM), "
								+ "hmm2 (second-order HMM).").create());
		options.addOption(OptionBuilder
				.withLongOpt("alg")
				.withArgName("perc | loss | afworse | tobetter | dual")
				.hasArg()
				.withDescription(
						"The training algorithm: "
								+ "perc (ordinary perceptron), "
								+ "loss (Loss-augmented perceptron), "
								+ "afworse (away-from-worse perceptron), "
								+ "tobetter (toward-better perceptron), "
								+ "dual (dual (kernelized) perceptron")
				.create());
		options.addOption(OptionBuilder
				.withLongOpt("kernel")
				.withArgName("poly2 | poly3 | poly4")
				.hasArg()
				.withDescription(
						"Kernel function: "
								+ "poly2 (2-degree polynomial function), "
								+ "poly3 (3-degree polynomial function), "
								+ "poly4 (4-degree polynomial function)")
				.create());
		// options.addOption(OptionBuilder
		// .withLongOpt("kcache")
		// .withDescription(
		// "Indicate the use of the kernel function cache")
		// .create());
		options.addOption(OptionBuilder
				.withLongOpt("distill")
				.withDescription(
						"Turn on the distillation procedure to minimize "
								+ "the number of support vectors on dual algorithms")
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

		options.addOption(OptionBuilder.withLongOpt("hashseed")
				.withArgName("seed").hasArg()
				.withDescription("Seed for the hash-based encodings.").create());

		options.addOption(OptionBuilder
				.withLongOpt("murmur3")
				.withArgName("size")
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

		/*
		 * Get the options given in the command-line or the corresponding
		 * default values.
		 */
		String[] inputCorpusFileNames = cmdLine.getOptionValues("incorpus");
		String additionalCorpusFileName = cmdLine.getOptionValue("inadd");
		String modelFileName = cmdLine.getOptionValue("model");
		int numEpochs = Integer.parseInt(cmdLine.getOptionValue("numepochs",
				"10"));
		double learningRate = Double.parseDouble(cmdLine.getOptionValue(
				"learnrate", "1"));
		String defaultLabel = cmdLine.getOptionValue("defstate", "0");
		String nullLabel = cmdLine.getOptionValue("nullstate", defaultLabel);
		String testCorpusFileName = cmdLine.getOptionValue("testcorpus");
		boolean evalPerEpoch = cmdLine.hasOption("perepoch");
		String labels = cmdLine.getOptionValue("labels");
		String encodingFile = cmdLine.getOptionValue("encoding");
		String hashSeed = cmdLine.getOptionValue("hashseed");
		String murmur = cmdLine.getOptionValue("murmur");
		String murmur3 = cmdLine.getOptionValue("murmur3");
		String murmur2 = cmdLine.getOptionValue("murmur2");
		String lookup3 = cmdLine.getOptionValue("lookup3");
		String javaHashSizeStr = cmdLine.getOptionValue("javahash");
		String tagsetFileName = cmdLine.getOptionValue("tagset");
		String nonAnnotatedLabel = cmdLine.getOptionValue("nonannlabel");
		Double reportProgressRate = Double.parseDouble(cmdLine.getOptionValue(
				"progress", "0.1"));
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

		SequenceDataset inputCorpusA = null;
		SequenceDataset inputCorpusB = null;
		double weightAdditionalCorpus = -1d;
		double weightStep = -1d;
		FeatureEncoding<String> featureEncoding = null;
		FeatureEncoding<String> additionalFeatureEncoding = null;
		StringMapEncoding stateEncoding = null;
		try {

			LOG.info("Creating/loading encoding...");

			// Create (or load) the feature value encoding.
			if (encodingFile != null) {

				/*
				 * Load a map-based encoding from the given file. Thus, the
				 * feature values present in this file will be encoded
				 * unambiguously but any unknown value will be ignored.
				 */
				featureEncoding = new StringMapEncoding(encodingFile);

			}

			/*
			 * Additional feature encoding (or the only one, if a fixed encoding
			 * file is not given).
			 */
			if (murmur != null) {

				// Create a feature encoding based on the Murmur3 hash function.
				int size = parseValueDirectOrBits(murmur);
				if (hashSeed == null)
					additionalFeatureEncoding = new Murmur3Encoding(size);
				else
					additionalFeatureEncoding = new Murmur3Encoding(size,
							Integer.parseInt(hashSeed));

			} else if (murmur3 != null) {

				// Create a feature encoding based on the Murmur3 hash function.
				int size = parseValueDirectOrBits(murmur3);
				if (hashSeed == null)
					additionalFeatureEncoding = new Murmur3Encoding(size);
				else
					additionalFeatureEncoding = new Murmur3Encoding(size,
							Integer.parseInt(hashSeed));

			} else if (murmur2 != null) {

				// Create a feature encoding based on the Murmur2 hash function.
				int size = parseValueDirectOrBits(murmur2);
				if (hashSeed == null)
					additionalFeatureEncoding = new Murmur2Encoding(size);
				else
					additionalFeatureEncoding = new Murmur2Encoding(size,
							Integer.parseInt(hashSeed));

			} else if (lookup3 != null) {

				// Create a feature encoding based on the Lookup3 hash function.
				int size = parseValueDirectOrBits(lookup3);
				if (hashSeed == null)
					additionalFeatureEncoding = new Lookup3Encoding(size);
				else
					additionalFeatureEncoding = new Lookup3Encoding(size,
							Integer.parseInt(hashSeed));

			} else if (javaHashSizeStr != null) {

				// Create a feature encoding based on the Java hash function.
				additionalFeatureEncoding = new JavaHashCodeEncoding(
						parseValueDirectOrBits(javaHashSizeStr));

			}

			if (featureEncoding == null) {

				if (additionalFeatureEncoding == null)
					/*
					 * No encoding given by the user. Create an empty and
					 * flexible feature encoding that will encode unambiguously
					 * all feature values. If the training dataset is big, this
					 * may not fit in memory.
					 */
					featureEncoding = new StringMapEncoding();
				else
					// Only one feature encoding given.
					featureEncoding = additionalFeatureEncoding;

			} else if (additionalFeatureEncoding != null)
				/*
				 * The user specified two encodings. Combine them in one hybrid
				 * encoding.
				 */
				featureEncoding = new HybridStringEncoding(featureEncoding,
						additionalFeatureEncoding);

			LOG.info("Feature encoding: "
					+ featureEncoding.getClass().getSimpleName());

			// Create or load the state label encoding.
			if (labels != null)
				// State set given in the command-line.
				stateEncoding = new StringMapEncoding(labels.split(","));
			else if (tagsetFileName != null)
				// State set given in a file.
				stateEncoding = new StringMapEncoding(tagsetFileName);
			else
				/*
				 * State set automatically retrieved from training data (codes
				 * depend on order of appereance of the labels).
				 */
				stateEncoding = new StringMapEncoding();

			LOG.info("Loading input corpus...");

			// Get the list of input paths and concatenate the corpora in them.
			inputCorpusA = new SequenceDataset(featureEncoding, stateEncoding,
					nonAnnotatedLabel, true);
			inputCorpusA
					.setSkipCompletelyNonAnnotatedExamples(skipCompletelyNonAnnotatedExamples);

			// Load the first data file, which can be the standard input.
			if (inputCorpusFileNames[0].equals("stdin"))
				inputCorpusA.load(System.in);
			else
				inputCorpusA.load(inputCorpusFileNames[0]);

			// Load other data files.
			for (int idxFile = 1; idxFile < inputCorpusFileNames.length; ++idxFile) {
				SequenceDataset other = new SequenceDataset(
						inputCorpusFileNames[idxFile], featureEncoding,
						stateEncoding, nonAnnotatedLabel,
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

				inputCorpusB = new SequenceDataset(additionalCorpusFileName,
						featureEncoding, stateEncoding, nonAnnotatedLabel,
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

		LOG.info("Feature encoding size: " + featureEncoding.size());
		LOG.info("Tagset size: " + stateEncoding.size());

		// Algorithm type.
		AlgorithmType algType = null;
		String algTypeStr = cmdLine.getOptionValue("alg", "perc");
		if (algTypeStr.equals("perc"))
			algType = AlgorithmType.PERCEPTRON;
		else if (algTypeStr.equals("loss"))
			algType = AlgorithmType.LOSS_PERCEPTRON;
		else if (algTypeStr.equals("afworse"))
			algType = AlgorithmType.AWAY_FROM_WORSE_PERCEPTRON;
		else if (algTypeStr.equals("tobetter"))
			algType = AlgorithmType.TOWARD_BETTER_PERCEPTRON;
		else if (algTypeStr.equals("dual"))
			algType = AlgorithmType.DUAL_PERCEPTRON;
		else {
			System.err.println("Unknown algorithm: " + algTypeStr);
			System.exit(1);
		}

		// Kernel.
		int polyKernelExponent = 1;
		String kernel = cmdLine.getOptionValue("kernel");
		if (kernel != null) {
			if (algType != AlgorithmType.DUAL_PERCEPTRON) {
				LOG.error("kernel=? requires alg=dual");
				System.exit(1);
			}

			if (kernel.equals("poly1"))
				polyKernelExponent = 1;
			else if (kernel.equals("poly2"))
				polyKernelExponent = 2;
			else if (kernel.equals("poly3"))
				polyKernelExponent = 3;
			else if (kernel.equals("poly4"))
				polyKernelExponent = 4;
			else {
				LOG.error("kernel=" + kernel + " is not a valid value");
				System.exit(1);
			}
		}

		// // Kernel function cache.
		// boolean kernelCache = cmdLine.hasOption("kcache");
		// if (kernelCache) {
		// if (algType != AlgorithmType.DUAL_PERCEPTRON) {
		// LOG.error("kcache requires alg=dual");
		// System.exit(1);
		// }
		// }

		// Structure.
		LOG.info("Allocating initial model...");
		Model model = null;
		Inference inference = null;
		String structure = cmdLine.getOptionValue("structure", "hmm");
		if (structure.equals("hmm")) {

			// Ordinary Viterbi-based inference algorithm.
			inference = new ViterbiInference(inputCorpusA.getStateEncoding()
					.put(defaultLabel));

			if (algType != AlgorithmType.DUAL_PERCEPTRON) {
				// Ordinary HMM model.
				// TODO test

				/*
				 * model = new
				 * AveragedArrayHmm(inputCorpusA.getNumberOfStates(),
				 * inputCorpusA.getNumberOfSymbols());
				 */

				model = new AveragedMapHmm(inputCorpusA.getNumberOfStates(),
						inputCorpusA.getNumberOfSymbols());

			} else {
				// Dual HMM model.
				model = new DualHmm(inputCorpusA.getInputs(),
						inputCorpusA.getOutputs(),
						inputCorpusA.getNumberOfStates(), polyKernelExponent);
				// // Activate or deactivate kernel function cache.
				// ((DualHmm)
				// model).setActivateKernelFunctionCache(kernelCache);
			}

		} else if (structure.equals("hmm2")) {

			if (algType == AlgorithmType.DUAL_PERCEPTRON) {
				// Dual 2nd-order HMM has not been implemented yet.
				System.err
						.println("alg=dual is not compatible with structure=hmm2");
				System.exit(1);
			}

			// 2nd order Viterbi-based inference algorithm.
			inference = new Viterbi2ndOrderInference(inputCorpusA
					.getStateEncoding().put(defaultLabel));

			// 2nd order HMM model.
			model = new AveragedArrayHmm2ndOrder(
					inputCorpusA.getNumberOfStates(),
					inputCorpusA.getNumberOfSymbols());

		} else {
			System.err.println("Unknown structure: " + structure);
			System.exit(1);
		}

		// Parse the task type option.
		TaskType taskType = null;
		String taskTypeStr = cmdLine.getOptionValue("task", "iob");
		if (taskTypeStr.equals("iob"))
			taskType = TaskType.IOB;
		else if (taskTypeStr.equals("token"))
			taskType = TaskType.TOKEN;
		else {
			System.err.println("Unknown task type: " + taskTypeStr);
			System.exit(1);
		}

		EntityF1Evaluation eval = null;
		switch (taskType) {
		case IOB:
			eval = new IobChunkEvaluation(inputCorpusA.getStateEncoding(),
					nullLabel);
			break;
		case TOKEN:
			eval = new LabeledTokenEvaluation(inputCorpusA.getStateEncoding());
			break;
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
			System.exit(1);
		}

		// Create the chosen algorithm.
		Perceptron alg = null;
		switch (algType) {

		/*
		 * Ordinary Perceptron implementation (Collins'): does not consider
		 * customized loss functions.
		 */
		case PERCEPTRON:
			alg = new Perceptron(inference, model, numEpochs, learningRate,
					true, averageWeights, learningRateUpdateStrategy);
			break;

		/*
		 * Loss-augumented implementation: considers customized loss function
		 * (per-token misclassification loss).
		 */
		case LOSS_PERCEPTRON:
			if (lossNonAnnotatedWeightStr == null)
				alg = new LossAugmentedPerceptron(inference, model, numEpochs,
						learningRate, lossWeight, true, averageWeights,
						learningRateUpdateStrategy);
			else
				alg = new LossAugmentedPerceptron(inference, model, numEpochs,
						learningRate, lossWeight,
						Double.parseDouble(lossNonAnnotatedWeightStr),
						lossNonAnnotatedWeightInc, true, averageWeights,
						learningRateUpdateStrategy);
			break;

		/*
		 * Away-from-worse perceptron implementation.
		 */
		case AWAY_FROM_WORSE_PERCEPTRON:
			if (lossNonAnnotatedWeightStr == null)
				alg = new AwayFromWorsePerceptron(inference, model, numEpochs,
						learningRate, lossWeight, true, averageWeights,
						learningRateUpdateStrategy);
			else
				alg = new AwayFromWorsePerceptron(inference, model, numEpochs,
						learningRate, lossWeight,
						Double.parseDouble(lossNonAnnotatedWeightStr),
						lossNonAnnotatedWeightInc, true, averageWeights,
						learningRateUpdateStrategy);
			break;

		/*
		 * Toward-better perceptron implementation.
		 */
		case TOWARD_BETTER_PERCEPTRON:
			if (lossNonAnnotatedWeightStr == null)
				alg = new TowardBetterPerceptron(inference, model, numEpochs,
						learningRate, lossWeight, true, averageWeights,
						learningRateUpdateStrategy);
			else
				alg = new TowardBetterPerceptron(inference, model, numEpochs,
						learningRate, lossWeight,
						Double.parseDouble(lossNonAnnotatedWeightStr),
						lossNonAnnotatedWeightInc, true, averageWeights,
						learningRateUpdateStrategy);
			break;

		/*
		 * Dual (kernelized) loss-augumented implementation: considers
		 * customized loss function (per-token misclassification loss) and uses
		 * a dual representation that allows kernel functions.
		 */
		case DUAL_PERCEPTRON:
			if (lossNonAnnotatedWeightStr == null)
				alg = new DualLossAugmentedPerceptron(inference,
						(DualModel) model, numEpochs, learningRate, lossWeight,
						true, averageWeights, learningRateUpdateStrategy);
			else
				alg = new DualLossAugmentedPerceptron(inference,
						(DualModel) model, numEpochs, learningRate, lossWeight,
						Double.parseDouble(lossNonAnnotatedWeightStr),
						lossNonAnnotatedWeightInc, true, averageWeights,
						learningRateUpdateStrategy);

			// Sort example features to speedup kernel functions.
			LOG.info("Sorting feature arrays...");
			inputCorpusA.sortFeatureValues();

			break;

		}

		if (nonAnnotatedLabel != null) {
			/*
			 * Signal the presence of partially-labeled examples to the
			 * algorithm.
			 */
			alg.setPartiallyAnnotatedExamples(true);
		}

		if (seedStr != null)
			// User provided seed to random number generator.
			alg.setSeed(Long.parseLong(seedStr));

		if (reportProgressRate != null)
			// Progress report rate.
			alg.setReportProgressRate(reportProgressRate);

		// Activate distillation process if required.
		if (cmdLine.hasOption("distill")) {
			if (algType == AlgorithmType.DUAL_PERCEPTRON) {
				((DualLossAugmentedPerceptron) alg).setDistill(true);
			} else {
				System.err.println("Option distill requires alg=dual");
				System.exit(1);
			}
		}

		// Ignore features not seen in the training corpus.
		inputCorpusA.getFeatureEncoding().setReadOnly(true);
		inputCorpusA.getStateEncoding().setReadOnly(true);

		// Evaluation after each training epoch.
		if (testCorpusFileName != null && evalPerEpoch) {
			try {

				LOG.info("Loading and preparing test data...");
				SequenceDataset testset = new SequenceDataset(
						testCorpusFileName, inputCorpusA.getFeatureEncoding(),
						inputCorpusA.getStateEncoding());

				if (algType == AlgorithmType.DUAL_PERCEPTRON) {
					// Sort example features to speedup kernel functions.
					LOG.info("Sorting test feature arrays...");
					testset.sortFeatureValues();
				}

				if (normalizeInput)
					// Normalize the input structures.
					testset.normalizeInputStructures(testset
							.getMaxNumberOfEmissionFeatures());

				alg.setListener(new EvaluateModelListener(eval, testset
						.getInputs(), testset.getOutputs(), inputCorpusA
						.getStateEncoding(), nullLabel, averageWeights,
						algType == AlgorithmType.DUAL_PERCEPTRON));

			} catch (Exception e) {
				LOG.error("Loading testset " + testCorpusFileName, e);
				System.exit(1);
			}
		} else {
			alg.setListener(new EvaluateModelListener(eval, null, null, null,
					null, false, algType == AlgorithmType.DUAL_PERCEPTRON));
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
			alg.train(inputCorpusA.getInputs(), inputCorpusA.getOutputs());
		} else {
			// Train on two datasets.
			if (weightAdditionalCorpus < 0d)
				/*
				 * If no different weight was given for the B dataset, then use
				 * a weight proportional to the sizes of the datasets.
				 */
				weightAdditionalCorpus = ((double) inputCorpusB
						.getNumberOfExamples())
						/ (inputCorpusA.getNumberOfExamples() + inputCorpusB
								.getNumberOfExamples());
			alg.train(inputCorpusA.getInputs(), inputCorpusA.getOutputs(),
					1d - weightAdditionalCorpus, weightStep,
					inputCorpusB.getInputs(), inputCorpusB.getOutputs());
		}

		// Evaluation only for the final model.
		if (testCorpusFileName != null && !evalPerEpoch) {
			try {

				LOG.info("Loading and preparing test data...");
				SequenceDataset testset = new SequenceDataset(
						testCorpusFileName, inputCorpusA.getFeatureEncoding(),
						inputCorpusA.getStateEncoding());

				if (algType == AlgorithmType.DUAL_PERCEPTRON) {
					// Sort example features to speedup kernel functions.
					LOG.info("Sorting test feature arrays...");
					testset.sortFeatureValues();
				}

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

				// Fill the list of predicted outputs.
				for (int idx = 0; idx < inputs.length; ++idx)
					// Predict (tag the output sequence).
					inference.inference(model, inputs[idx], predicteds[idx]);

				// Evaluate the sequences.
				Map<String, F1Measure> results = eval.evaluateExamples(inputs,
						outputs, predicteds);

				// Write results (precision, recall and F-1) per class.
				printF1Results("Final performance:", results);

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
				model.save(ps, inputCorpusA.getFeatureEncoding(),
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
	public static int parseValueDirectOrBits(String valStr) {
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
	 * Print the given result set and title.
	 * 
	 * @param title
	 * @param results
	 */
	private static void printF1Results(String title,
			Map<String, F1Measure> results) {

		// Title and header.
		System.out.println("\n" + title + "\n");
		System.out.println("|+");
		System.out
				.println("! Class !! P !! R !! F !! Total (TP+FN) !! Retrieved (TP+FP) !! Correct (TP)");

		// Per-class results.
		for (Entry<String, F1Measure> res : results.entrySet()) {
			String label = res.getKey();
			if (label.equals("overall"))
				continue;
			F1Measure f1 = res.getValue();
			if (f1 == null)
				continue;
			System.out.println("|-");
			System.out.println(String.format(
					"| %s || %.2f ||  %.2f || %.2f || %d || %d || %d", label,
					100 * f1.getPrecision(), 100 * f1.getRecall(),
					100 * f1.getF1(), f1.getNumObjects(), f1.getNumRetrieved(),
					f1.getNumCorrectlyRetrieved()));
		}

		// Overall result.
		F1Measure f1 = results.get("overall");
		if (f1 != null) {
			System.out.println("|-");
			System.out.println(String.format(
					"| %s || %.2f || %.2f || %.2f || %d || %d || %d",
					"overall", 100 * f1.getPrecision(), 100 * f1.getRecall(),
					100 * f1.getF1(), f1.getNumObjects(), f1.getNumRetrieved(),
					f1.getNumCorrectlyRetrieved()));
		}

		// Footer.
		System.out.println();

	}

	/**
	 * Training listener to evaluate models after each iteration.
	 * 
	 * @author eraldof
	 * 
	 */
	private static class EvaluateModelListener implements TrainingListener {

		private EntityF1Evaluation eval;

		private SequenceInput[] inputs;

		private SequenceOutput[] outputs;

		private SequenceOutput[] predicteds;

		private boolean averageWeights;

		private boolean dual;

		public EvaluateModelListener(EntityF1Evaluation eval,
				SequenceInput[] inputs, SequenceOutput[] outputs,
				FeatureEncoding<String> stateEncoding, String nullLabel,
				boolean averageWeights, boolean dual) {
			this.inputs = inputs;
			this.outputs = outputs;
			this.eval = eval;
			this.averageWeights = averageWeights;
			this.dual = dual;
			if (inputs != null) {
				this.predicteds = new SequenceOutput[inputs.length];
				// Allocate output sequences for predictions.
				for (int idx = 0; idx < inputs.length; ++idx)
					predicteds[idx] = (SequenceOutput) inputs[idx]
							.createOutput();
			}
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

			if (inputs == null)
				return true;

			if (averageWeights) {
				try {

					// Clone the current model to average it, if necessary.
					hmm = (Model) hmm.clone();

					/*
					 * The averaged perceptron averages the final model only in
					 * the end of the training process, hence we need to average
					 * the temporary model here in order to have a better
					 * picture of its current (intermediary) performance.
					 */
					hmm.average(iteration);

				} catch (CloneNotSupportedException e) {
					LOG.error("Cloning current model on epoch " + epoch
							+ " and iteration " + iteration, e);
					return true;
				}
			}

			// Fill the list of predicted outputs.
			for (int idx = 0; idx < inputs.length; ++idx)
				// Predict (tag the output sequence).
				inferenceImpl.inference(hmm, inputs[idx], predicteds[idx]);

			// Evaluate the sequences.
			Map<String, F1Measure> results = eval.evaluateExamples(inputs,
					outputs, predicteds);

			// Write results (precision, recall and F-1) per class.
			printF1Results("Performance after epoch " + epoch + ":", results);

			return true;
		}

		@Override
		public void progressReport(Inference impl, Model curModel, int epoch,
				double loss, int iteration) {
			if (!dual)
				return;
			DualHmm dualHmm = (DualHmm) curModel;
			LOG.info(String.format(
					"Iteration: %d | Loss: %f | # Exs w/ SVs: %d | # SVs: %d",
					iteration, loss,
					dualHmm.getNumberOfExamplesWithSupportVector(),
					dualHmm.getNumberOfSupportVectors()));
		}
	}
}
