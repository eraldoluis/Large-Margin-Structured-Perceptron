package br.pucrio.inf.learn.structlearning.discriminative.driver;

import java.io.FileNotFoundException;
import java.io.PrintStream;
import java.util.Collection;
import java.util.Map;

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
import br.pucrio.inf.learn.structlearning.discriminative.application.dp.DPBasicModel;
import br.pucrio.inf.learn.structlearning.discriminative.application.dp.DPModel;
import br.pucrio.inf.learn.structlearning.discriminative.application.dp.DPTemplateEvolutionModel;
import br.pucrio.inf.learn.structlearning.discriminative.application.dp.DPTemplateModel;
import br.pucrio.inf.learn.structlearning.discriminative.application.dp.MaximumBranchingInference;
import br.pucrio.inf.learn.structlearning.discriminative.application.dp.data.DPBasicDataset;
import br.pucrio.inf.learn.structlearning.discriminative.application.dp.data.DPColumnDataset;
import br.pucrio.inf.learn.structlearning.discriminative.application.dp.data.DPDataset;
import br.pucrio.inf.learn.structlearning.discriminative.application.dp.data.DPInput;
import br.pucrio.inf.learn.structlearning.discriminative.application.dp.data.DPOutput;
import br.pucrio.inf.learn.structlearning.discriminative.application.dp.evaluation.DPEvaluation;
import br.pucrio.inf.learn.structlearning.discriminative.data.encoding.FeatureEncoding;
import br.pucrio.inf.learn.structlearning.discriminative.data.encoding.HybridStringEncoding;
import br.pucrio.inf.learn.structlearning.discriminative.data.encoding.JavaHashCodeEncoding;
import br.pucrio.inf.learn.structlearning.discriminative.data.encoding.Lookup3Encoding;
import br.pucrio.inf.learn.structlearning.discriminative.data.encoding.Murmur2Encoding;
import br.pucrio.inf.learn.structlearning.discriminative.data.encoding.Murmur3Encoding;
import br.pucrio.inf.learn.structlearning.discriminative.data.encoding.StringMapEncoding;
import br.pucrio.inf.learn.structlearning.discriminative.driver.Driver.Command;
import br.pucrio.inf.learn.structlearning.discriminative.evaluation.AccuracyEvaluation;
import br.pucrio.inf.learn.structlearning.discriminative.task.DualModel;
import br.pucrio.inf.learn.structlearning.discriminative.task.Inference;
import br.pucrio.inf.learn.structlearning.discriminative.task.Model;
import br.pucrio.inf.learn.util.CommandLineOptionsUtil;
import br.pucrio.inf.learn.util.DebugUtil;

/**
 * Driver to discriminatively train a dependency parser using perceptron-based
 * algorithms.
 * 
 * @author eraldo
 * 
 */
public class TrainDP implements Command {

	/**
	 * Logging object.
	 */
	private static final Log LOG = LogFactory.getLog(TrainDP.class);

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
		options.addOption(OptionBuilder.withLongOpt("train").isRequired()
				.withArgName("filename").hasArg()
				.withDescription("Training dataset file name.").create());
		options.addOption(OptionBuilder.withLongOpt("trainpunc")
				.withArgName("filename").hasArg()
				.withDescription("Punctuation file name for train dataset.")
				.create());
		options.addOption(OptionBuilder.withLongOpt("testpunc")
				.withArgName("filename").hasArg()
				.withDescription("Punctuation file name for test dataset.")
				.create());
		options.addOption(OptionBuilder
				.withLongOpt("templates")
				.withArgName("filename")
				.hasArg()
				.withDescription(
						"Feature templates file name. Implies that train and "
								+ "test must be column-format datasets.")
				.create());
		options.addOption(OptionBuilder.withLongOpt("index")
				.withDescription("Activate inverted index.").create());
		options.addOption(OptionBuilder
				.withLongOpt("testexplicit")
				.withDescription(
						"Activate explicit features lists for test "
								+ "dataset.").create());
		options.addOption(OptionBuilder
				.withLongOpt("serial")
				.withDescription(
						"Load the training dataset from a "
								+ "serialized (binary) file.").create());
		options.addOption(OptionBuilder
				.withLongOpt("model")
				.hasArg()
				.withArgName("filename")
				.withDescription(
						"Name of the file to save the resulting model.")
				.create());
		options.addOption(OptionBuilder
				.withLongOpt("numepochs")
				.withArgName("integer")
				.hasArg()
				.withDescription(
						"Number of epochs: how many iterations over the"
								+ " training set.").create());
		options.addOption(OptionBuilder.withLongOpt("learnrate")
				.withArgName("[0:1]").hasArg()
				.withDescription("Learning rate used in the updates.").create());
		options.addOption(OptionBuilder
				.withLongOpt("encoding")
				.withArgName("filename")
				.hasArg()
				.withDescription(
						"Filename that contains a list of considered feature"
								+ " values. Any feature value not present in"
								+ " this file is ignored.").create());
		options.addOption(OptionBuilder
				.withLongOpt("minfreq")
				.withArgName("integer")
				.hasArg()
				.withDescription(
						"Minimum frequency of feature values in the encoding "
								+ "file used to cutoff low frequent values.")
				.create());
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
		options.addOption(OptionBuilder.withLongOpt("test")
				.withArgName("filename").hasArg()
				.withDescription("Test corpus file name.").create());
		options.addOption(OptionBuilder
				.withLongOpt("perepoch")
				.withDescription(
						"The evaluation on the test corpus will "
								+ "be performed after each training epoch.")
				.create());
		options.addOption(OptionBuilder
				.withLongOpt("progress")
				.withArgName("rate of examples")
				.hasArg()
				.withDescription(
						"Rate to report the training progress within each"
								+ " epoch.").create());
		options.addOption(OptionBuilder.withLongOpt("seed")
				.withArgName("integer").hasArg()
				.withDescription("Random number generator seed.").create());
		options.addOption(OptionBuilder
				.withLongOpt("lossweight")
				.withArgName("double")
				.hasArg()
				.withDescription(
						"Weight of the loss term in the inference objective"
								+ " function.").create());
		options.addOption(OptionBuilder
				.withLongOpt("lossweightinc")
				.withArgName("double")
				.hasArg()
				.withDescription(
						"Increment in the loss weight after each epoch.")
				.create());
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
		options.addOption(OptionBuilder.withLongOpt("debug")
				.withDescription("Print debug information.").create());

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
		String[] inputCorpusFileNames = cmdLine.getOptionValues("train");
		String puncFileNameTrain = cmdLine.getOptionValue("trainpunc");
		String puncFileNameTest = cmdLine.getOptionValue("testpunc");
		String templatesFileName = cmdLine.getOptionValue("templates");
		boolean hasInvertedIndex = cmdLine.hasOption("index");
		String modelFileName = cmdLine.getOptionValue("model");
		int numEpochs = Integer.parseInt(cmdLine.getOptionValue("numepochs",
				"10"));
		double learningRate = Double.parseDouble(cmdLine.getOptionValue(
				"learnrate", "1"));
		String testCorpusFileName = cmdLine.getOptionValue("test");
		boolean evalPerEpoch = cmdLine.hasOption("perepoch");
		String encodingFile = cmdLine.getOptionValue("encoding");
		String minFreqStr = cmdLine.getOptionValue("minfreq");
		String hashSeed = cmdLine.getOptionValue("hashseed");
		String murmur = cmdLine.getOptionValue("murmur");
		String murmur3 = cmdLine.getOptionValue("murmur3");
		String murmur2 = cmdLine.getOptionValue("murmur2");
		String lookup3 = cmdLine.getOptionValue("lookup3");
		String javaHashSizeStr = cmdLine.getOptionValue("javahash");
		Double reportProgressRate = Double.parseDouble(cmdLine.getOptionValue(
				"progress", "0.1"));
		String seedStr = cmdLine.getOptionValue("seed");
		double lossWeight = Double.parseDouble(cmdLine.getOptionValue(
				"lossweight", "0d"));
		double lossWeightInc = Double.parseDouble(cmdLine.getOptionValue(
				"lossweightinc", "0"));
		boolean averageWeights = !cmdLine.hasOption("noavg");
		String lrUpdateStrategy = cmdLine.getOptionValue("lrupdate");
		boolean debug = cmdLine.hasOption("debug");
		boolean serialDatasets = cmdLine.hasOption("serial");

		DPDataset inDataset = null;
		int sizeEncoding = -1;
		FeatureEncoding<String> featureEncoding = null;
		FeatureEncoding<String> additionalFeatureEncoding = null;
		try {

			if (serialDatasets && encodingFile != null) {
				// Only load the size of the encoding.
				LOG.info("Loading encoding size (serialized datasets)...");
				sizeEncoding = new StringMapEncoding().loadSize(encodingFile);
			} else {
				// Create (or load) the feature value encoding.
				if (encodingFile != null) {

					if (minFreqStr == null) {
						/*
						 * Load a map-based encoding from the given file. Thus,
						 * the feature values present in this file will be
						 * encoded unambiguously but any unknown value will be
						 * ignored.
						 */
						LOG.info("Loading encoding file...");
						featureEncoding = new StringMapEncoding(encodingFile);
					} else {
						/*
						 * Load map-based encoding from the given file and
						 * filter out low frequent feature values according to
						 * feature frequencies given in the file.
						 */
						LOG.info("Loading encoding file...");
						int minFreq = Integer.parseInt(minFreqStr);
						featureEncoding = new StringMapEncoding(encodingFile,
								minFreq);
					}

				} else if (minFreqStr != null) {
					LOG.error("minfreq=? only works together with option encoding");
					System.exit(1);
				}

				/*
				 * Additional feature encoding (or the only one, if a fixed
				 * encoding file is not given).
				 */
				if (murmur != null) {

					// Create a feature encoding based on the Murmur3 hash
					// function.
					int size = parseValueDirectOrBits(murmur);
					if (hashSeed == null)
						additionalFeatureEncoding = new Murmur3Encoding(size);
					else
						additionalFeatureEncoding = new Murmur3Encoding(size,
								Integer.parseInt(hashSeed));

				} else if (murmur3 != null) {

					// Create a feature encoding based on the Murmur3 hash
					// function.
					int size = parseValueDirectOrBits(murmur3);
					if (hashSeed == null)
						additionalFeatureEncoding = new Murmur3Encoding(size);
					else
						additionalFeatureEncoding = new Murmur3Encoding(size,
								Integer.parseInt(hashSeed));

				} else if (murmur2 != null) {

					// Create a feature encoding based on the Murmur2 hash
					// function.
					int size = parseValueDirectOrBits(murmur2);
					if (hashSeed == null)
						additionalFeatureEncoding = new Murmur2Encoding(size);
					else
						additionalFeatureEncoding = new Murmur2Encoding(size,
								Integer.parseInt(hashSeed));

				} else if (lookup3 != null) {

					// Create a feature encoding based on the Lookup3 hash
					// function.
					int size = parseValueDirectOrBits(lookup3);
					if (hashSeed == null)
						additionalFeatureEncoding = new Lookup3Encoding(size);
					else
						additionalFeatureEncoding = new Lookup3Encoding(size,
								Integer.parseInt(hashSeed));

				} else if (javaHashSizeStr != null) {

					// Create a feature encoding based on the Java hash
					// function.
					additionalFeatureEncoding = new JavaHashCodeEncoding(
							parseValueDirectOrBits(javaHashSizeStr));

				}

				if (featureEncoding == null) {

					if (additionalFeatureEncoding == null)
						/*
						 * No encoding given by the user. Create an empty and
						 * flexible feature encoding that will encode
						 * unambiguously all feature values. If the training
						 * dataset is big, this may not fit in memory.
						 */
						featureEncoding = new StringMapEncoding();
					else
						// Only one feature encoding given.
						featureEncoding = additionalFeatureEncoding;

				} else if (additionalFeatureEncoding != null)
					/*
					 * The user specified two encodings. Combine them in one
					 * hybrid encoding.
					 */
					featureEncoding = new HybridStringEncoding(featureEncoding,
							additionalFeatureEncoding);

				LOG.info("Feature encoding: "
						+ featureEncoding.getClass().getSimpleName());
			}

			if (templatesFileName == null) {
				inDataset = new DPBasicDataset(featureEncoding);
				if (serialDatasets) {
					// Load a serialized dataset.
					LOG.info("Loading input corpus from a serialized file...");
					inDataset.deserialize(inputCorpusFileNames[0]);
				} else {
					// Load from a textual file.
					LOG.info("Loading input corpus...");
					if (inDataset.equals("stdin"))
						inDataset.load(System.in);
					else
						inDataset.load(inputCorpusFileNames[0]);
				}
			} else {
				// Load templates and edge corpus.
				LOG.info("Loading edge corpus...");
				inDataset = new DPColumnDataset(featureEncoding,
						(Collection<String>) null);
				if (puncFileNameTrain != null)
					((DPColumnDataset) inDataset)
							.setFileNamePunc(puncFileNameTrain);
				inDataset.load(inputCorpusFileNames[0]);
				LOG.info("Loading templates and generating features...");
				((DPColumnDataset) inDataset).loadTemplates(templatesFileName,
						true);
			}

		} catch (Exception e) {
			LOG.error("Parsing command-line options", e);
			System.exit(1);
		}

		if (sizeEncoding == -1)
			LOG.info("Feature encoding size: " + featureEncoding.size());
		else
			LOG.info("Feature encoding size: " + sizeEncoding);

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
			LOG.error("Unknown algorithm: " + algTypeStr);
			System.exit(1);
		}

		if (sizeEncoding == -1)
			// Serialized datasets.
			sizeEncoding = featureEncoding.size();
		Model model;
		Inference inference;
		if (templatesFileName == null) {
			if (hasInvertedIndex) {
				LOG.error("Option --index requires --templates=<file>");
				System.exit(1);
			}
			// Explicit-features model.
			LOG.info("Allocating initial model...");
			model = new DPBasicModel();
		} else if (hasInvertedIndex) {
			LOG.info("Allocating initial model...");
			model = new DPTemplateModel();
			// Template-based model with inverted index.
			LOG.info("Creating inverted index...");
			((DPColumnDataset) inDataset).createInvertedIndex();
			// ((DPTemplateModel) model).init((DPColumnDataset) inDataset);
		} else {
			// Template-based model.
			LOG.info("Allocating initial model...");
			// model = new DPTemplateModel(templates[0]);
			// ((DPTemplateModel) model).init((DPColumnDataset) inDataset);
			model = new DPTemplateEvolutionModel();
		}

		// Inference algorithm.
		inference = new MaximumBranchingInference(
				inDataset.getMaxNumberOfTokens());

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
			if (lossWeightInc != 0d) {
				LOG.error("lossweightinc is not compatible with --alg=perc");
				System.exit(1);
			}
			break;

		/*
		 * Loss-augumented implementation: considers customized loss function
		 * (per-token misclassification loss).
		 */
		case LOSS_PERCEPTRON:
			alg = new LossAugmentedPerceptron(inference, model, numEpochs,
					learningRate, lossWeight, true, averageWeights,
					learningRateUpdateStrategy);
			if (lossWeightInc != 0d)
				((LossAugmentedPerceptron) alg)
						.setLossWeightIncrement(lossWeightInc);
			break;

		/*
		 * Away-from-worse perceptron implementation.
		 */
		case AWAY_FROM_WORSE_PERCEPTRON:
			alg = new AwayFromWorsePerceptron(inference, model, numEpochs,
					learningRate, lossWeight, true, averageWeights,
					learningRateUpdateStrategy);
			if (lossWeightInc != 0d)
				((LossAugmentedPerceptron) alg)
						.setLossWeightIncrement(lossWeightInc);
			break;

		/*
		 * Toward-better perceptron implementation.
		 */
		case TOWARD_BETTER_PERCEPTRON:
			alg = new TowardBetterPerceptron(inference, model, numEpochs,
					learningRate, lossWeight, true, averageWeights,
					learningRateUpdateStrategy);
			if (lossWeightInc != 0d)
				((LossAugmentedPerceptron) alg)
						.setLossWeightIncrement(lossWeightInc);
			break;

		/*
		 * Dual (kernelized) loss-augumented implementation: considers
		 * customized loss function (per-token misclassification loss) and uses
		 * a dual representation that allows kernel functions.
		 */
		case DUAL_PERCEPTRON:
			alg = new DualLossAugmentedPerceptron(inference, (DualModel) model,
					numEpochs, learningRate, lossWeight, true, averageWeights,
					learningRateUpdateStrategy);
			break;

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
		if (featureEncoding != null)
			featureEncoding.setReadOnly(true);

		// Evaluation method.
		DPEvaluation eval = new DPEvaluation(false);

		// Evaluation after each training epoch.
		boolean testExplicitFeatures = cmdLine.hasOption("testexplicit");
		if (testCorpusFileName != null && evalPerEpoch) {
			try {

				LOG.info("Loading and preparing test data...");
				DPDataset testset;
				if (templatesFileName == null)
					testset = new DPBasicDataset(inDataset.getFeatureEncoding());
				else {
					testset = new DPColumnDataset((DPColumnDataset) inDataset);
					if (puncFileNameTest != null)
						((DPColumnDataset) testset)
								.setFileNamePunc(puncFileNameTest);
				}

				if (serialDatasets)
					testset.deserialize(testCorpusFileName);
				else
					testset.load(testCorpusFileName);

				if (templatesFileName != null)
					((DPColumnDataset) testset).generateFeatures();

				alg.setListener(new EvaluateModelListener(eval, testset,
						averageWeights, testExplicitFeatures));

			} catch (Exception e) {
				LOG.error("Loading testset " + testCorpusFileName, e);
				System.exit(1);
			}
		} else {
			alg.setListener(new EvaluateModelListener(eval, null, false, false));
		}

		// Debug information.
		if (debug) {
			DebugUtil.featureEncoding = featureEncoding;
			DebugUtil.print = true;
		}

		LOG.info("Training model...");
		// Train model.
		alg.train(inDataset.getInputs(), inDataset.getOutputs());

		// Evaluation only for the final model.
		if (testCorpusFileName != null && !evalPerEpoch) {
			try {

				LOG.info("Loading and preparing test data...");
				DPDataset testset;
				if (templatesFileName == null)
					testset = new DPBasicDataset(inDataset.getFeatureEncoding());
				else {
					testset = new DPColumnDataset((DPColumnDataset) inDataset);
					if (puncFileNameTest != null)
						((DPColumnDataset) testset)
								.setFileNamePunc(puncFileNameTest);
				}

				if (serialDatasets)
					testset.deserialize(testCorpusFileName);
				else
					testset.load(testCorpusFileName);

				if (templatesFileName != null)
					((DPColumnDataset) testset).generateFeatures();

				// Allocate output sequences for predictions.
				DPInput[] inputs = testset.getInputs();
				DPOutput[] outputs = testset.getOutputs();
				DPOutput[] predicteds = new DPOutput[inputs.length];
				for (int idx = 0; idx < inputs.length; ++idx)
					predicteds[idx] = (DPOutput) inputs[idx].createOutput();

				// Fill the list of predicted outputs.
				for (int idx = 0; idx < inputs.length; ++idx)
					// Predict (tag the output sequence).
					inference.inference(model, inputs[idx], predicteds[idx]);

				// Evaluate the sequences.
				Map<String, Double> results = eval.evaluateExamples(inputs,
						outputs, predicteds);

				// Write results (precision, recall and F-1) per class.
				printAccuracyResults("Final performance:", results);

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
				/*
				 * model.save(ps, inputCorpusA.getFeatureEncoding(),
				 * inputCorpusA.getStateEncoding());
				 */
				ps.close();
			} catch (FileNotFoundException e) {
				LOG.error("Saving model " + modelFileName, e);
			}
		}

		LOG.info(String.format("# updated parameters: %d",
				((DPModel) model).getNumberOfUpdatedParameters()));

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
	private static void printAccuracyResults(String title,
			Map<String, Double> results) {

		// Title and header.
		System.out.println("\n" + title + "\n");
		System.out.println("|+");
		System.out.println("! Type !! Accuracy");

		// Per-class results.
		System.out.println("|-");
		System.out.println(String.format("| %s || %.4f ", "Average accuracy",
				100 * results.get("average")));
		System.out.println("|-");
		System.out.println(String.format("| %s || %.4f ",
				"Per-example accuracy", 100 * results.get("example")));

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

		private AccuracyEvaluation eval;

		private DPInput[] inputs;

		private DPOutput[] outputs;

		private DPOutput[] predicteds;

		private boolean averageWeights;

		private boolean explicitFeatures;

		public EvaluateModelListener(AccuracyEvaluation eval,
				DPDataset testset, boolean averageWeights,
				boolean explicitFeatures) {
			if (testset != null) {
				this.inputs = testset.getInputs();
				this.outputs = testset.getOutputs();
			}
			this.eval = eval;
			this.averageWeights = averageWeights;
			this.explicitFeatures = explicitFeatures;
			if (inputs != null) {
				this.predicteds = new DPOutput[inputs.length];
				// Allocate output sequences for predictions.
				for (int idx = 0; idx < inputs.length; ++idx)
					predicteds[idx] = (DPOutput) inputs[idx].createOutput();
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
		public boolean afterEpoch(Inference inferenceImpl, Model model,
				int epoch, double loss, int iteration) {

			if (inputs == null)
				return true;

			if (averageWeights || explicitFeatures) {
				try {
					// Clone the current model to average it, if necessary.
					model = (Model) model.clone();
				} catch (CloneNotSupportedException e) {
					LOG.error("Cloning current model on epoch " + epoch
							+ " and iteration " + iteration, e);
					return true;
				}
			}

			/*
			 * The averaged perceptron averages the final model only in the end
			 * of the training process, hence we need to average the temporary
			 * model here in order to have a better picture of its current
			 * (intermediary) performance.
			 */
			if (averageWeights)
				model.average(iteration);

			// Fill the list of predicted outputs.
			for (int idx = 0; idx < inputs.length; ++idx)
				// Predict (tag the output sequence).
				inferenceImpl.inference(model, inputs[idx], predicteds[idx]);

			// Evaluate the sequences.
			Map<String, Double> results = eval.evaluateExamples(inputs,
					outputs, predicteds);

			// Write results (precision, recall and F-1) per class.
			printAccuracyResults("Performance after epoch " + epoch + ":",
					results);

			return true;
		}

		@Override
		public void progressReport(Inference impl, Model curModel, int epoch,
				double loss, int iteration) {
		}
	}
}
