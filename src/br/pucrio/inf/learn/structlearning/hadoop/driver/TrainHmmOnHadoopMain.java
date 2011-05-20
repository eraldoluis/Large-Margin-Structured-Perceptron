package br.pucrio.inf.learn.structlearning.hadoop.driver;

import java.io.InputStream;

import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.OptionBuilder;
import org.apache.commons.cli.Options;
import org.apache.commons.cli.ParseException;
import org.apache.commons.cli.PosixParser;
import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.hadoop.conf.Configured;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.hdfs.DistributedFileSystem;
import org.apache.hadoop.io.MapWritable;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.lib.input.SequenceFileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.SequenceFileOutputFormat;
import org.apache.hadoop.util.Tool;

import br.pucrio.inf.learn.mr.data.HmmDistributionKey;
import br.pucrio.inf.learn.structlearning.data.FeatureEncoding;
import br.pucrio.inf.learn.structlearning.data.StringMapEncoding;
import br.pucrio.inf.learn.structlearning.hadoop.config.TrainHmmConfig;
import br.pucrio.inf.learn.structlearning.hadoop.mapper.TrainHmmMapper;
import br.pucrio.inf.learn.util.CommandLineOptionsUtil;

public class TrainHmmOnHadoopMain extends Configured implements Tool {

	private static final Log LOG = LogFactory
			.getLog(TrainHmmOnHadoopMain.class);

	@SuppressWarnings("static-access")
	public int run(String[] args) throws Exception {
		Options options = new Options();
		options.addOption(OptionBuilder.withLongOpt("incorpus").isRequired()
				.withArgName("input corpus").hasArg()
				.withDescription("Input corpus file name.").create('i'));
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
		options.addOption(OptionBuilder
				.withLongOpt("inadd")
				.withArgName("additional corpus[,weight[,step]]")
				.hasArg()
				.isRequired()
				.withDescription(
						"Additional corpus file name and "
								+ "an optional weight separated by comma and "
								+ "an weight step.").create());
		options.addOption(OptionBuilder
				.withLongOpt("model")
				.hasArg()
				.isRequired()
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
				.withLongOpt("encoding")
				.withArgName("feature values encoding file")
				.hasArg()
				.isRequired()
				.withDescription(
						"Filename that contains a list of considered feature"
								+ " values. Any feature value not present in"
								+ " this file is ignored.").create());
		options.addOption(OptionBuilder
				.withLongOpt("tagset")
				.withArgName("tagset file name")
				.hasArg()
				.isRequired()
				.withDescription(
						"Name of a file that contains the list of labels, one"
								+ " per line. This can be usefull to specify "
								+ "the preference order of state labels.")
				.create());
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
		String algTypeStr = cmdLine.getOptionValue("alg");
		String inputCorpusFileName = cmdLine.getOptionValue("incorpus");
		String additionalCorpusFileName = cmdLine.getOptionValue("inadd");
		String modelFileName = cmdLine.getOptionValue("model");
		String numEpochsStr = cmdLine.getOptionValue("numepochs");
		String learningRateStr = cmdLine.getOptionValue("learnrate");
		String defaultLabel = cmdLine.getOptionValue("defstate");
		String encodingFile = cmdLine.getOptionValue("encoding");
		String tagsetFileName = cmdLine.getOptionValue("tagset");
		String seedStr = cmdLine.getOptionValue("seed");
		String lossWeightStr = cmdLine.getOptionValue("lossweight");
		boolean nonAverageWeights = cmdLine.hasOption("noavg");
		String lossNonAnnotatedWeightStr = cmdLine
				.getOptionValue("lossnonlabeledweight");

		Job job = new Job(getConf(), getClass().getSimpleName());

		// Get the distributed file system (DFS).
		FileSystem dfs = DistributedFileSystem.get(job.getConfiguration());

		// Load the feature encoding from the DFS.
		InputStream is = dfs.open(new Path(encodingFile));
		FeatureEncoding<String> featureEncoding = new StringMapEncoding(is);
		is.close();

		// Load the feature encoding from the DFS.
		is = dfs.open(new Path(tagsetFileName));
		FeatureEncoding<String> stateEncoding = new StringMapEncoding(is);
		is.close();

		// Set the required options.
		// TODO TrainHmmConfig.setSmallCorpus(job, inputCorpusFileName);
		TrainHmmConfig.setNumberOfStates(job, stateEncoding.size());
		TrainHmmConfig.setNumberOfSymbols(job, featureEncoding.size());

		// Set the optional options.
		if (algTypeStr != null)
			TrainHmmConfig.setAlgorithm(job, algTypeStr);
		if (numEpochsStr != null)
			TrainHmmConfig.setNumberOfEpochs(job,
					Integer.parseInt(numEpochsStr));
		if (defaultLabel != null)
			TrainHmmConfig
					.setDefaultState(job, stateEncoding.put(defaultLabel));
		if (learningRateStr != null)
			TrainHmmConfig.setLearningRate(job,
					Float.parseFloat(learningRateStr));
		if (lossWeightStr != null)
			TrainHmmConfig.setLossWeight(job, Float.parseFloat(lossWeightStr));
		if (lossNonAnnotatedWeightStr != null)
			TrainHmmConfig.setNonAnnotatedLossWeight(job,
					Float.parseFloat(lossNonAnnotatedWeightStr));
		if (nonAverageWeights)
			TrainHmmConfig.setNonAverageWeights(job, nonAverageWeights);
		if (seedStr != null)
			TrainHmmConfig.setSeed(job, Integer.parseInt(seedStr));

		// Jar file to be distributed.
		job.setJarByClass(getClass());

		// Input and output paths.
		SequenceFileInputFormat.setInputPaths(job, additionalCorpusFileName);
		SequenceFileOutputFormat.setOutputPath(job, new Path(modelFileName));

		// Input format and mapper.
		job.setMapperClass(TrainHmmMapper.class);
		job.setInputFormatClass(SequenceFileInputFormat.class);

		// Intermediate classes and reducer.
		job.setMapOutputKeyClass(HmmDistributionKey.class);
		job.setMapOutputValueClass(MapWritable.class);
		// job.setReducerClass(null);

		// Output classes and format.
		job.setOutputKeyClass(HmmDistributionKey.class);
		job.setOutputValueClass(MapWritable.class);
		job.setOutputFormatClass(SequenceFileOutputFormat.class);

		// Submit job and wait for completion.
		job.submit();
		LOG.info("Job URL tracking: " + job.getTrackingURL());
		job.waitForCompletion(true);

		return 0;

	}

}
