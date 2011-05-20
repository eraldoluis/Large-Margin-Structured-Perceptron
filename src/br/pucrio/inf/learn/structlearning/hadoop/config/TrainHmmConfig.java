package br.pucrio.inf.learn.structlearning.hadoop.config;

import java.io.IOException;
import java.net.URI;
import java.net.URISyntaxException;
import java.util.Map.Entry;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.filecache.DistributedCache;
import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.MapWritable;
import org.apache.hadoop.io.Writable;
import org.apache.hadoop.mapreduce.JobContext;
import org.apache.hadoop.mapreduce.TaskInputOutputContext;

import br.pucrio.inf.learn.mr.data.HmmDistributionKey;
import br.pucrio.inf.learn.mr.data.HmmDistributionKey.DistributionType;
import br.pucrio.inf.learn.mr.util.MultipleSequenceFileReader;
import br.pucrio.inf.learn.structlearning.algorithm.StructuredAlgorithm;
import br.pucrio.inf.learn.structlearning.algorithm.perceptron.AwayFromWorsePerceptron;
import br.pucrio.inf.learn.structlearning.algorithm.perceptron.LossAugmentedPerceptron;
import br.pucrio.inf.learn.structlearning.algorithm.perceptron.Perceptron;
import br.pucrio.inf.learn.structlearning.algorithm.perceptron.Perceptron.LearnRateUpdateStrategy;
import br.pucrio.inf.learn.structlearning.algorithm.perceptron.TowardBetterPerceptron;
import br.pucrio.inf.learn.structlearning.application.sequence.AveragedArrayBasedHmm;
import br.pucrio.inf.learn.structlearning.application.sequence.Hmm;
import br.pucrio.inf.learn.structlearning.application.sequence.ViterbiInference;
import br.pucrio.inf.learn.structlearning.application.sequence.data.Dataset;
import br.pucrio.inf.learn.structlearning.application.sequence.data.DatasetException;
import br.pucrio.inf.learn.structlearning.driver.TrainHmmMain;

/**
 * Implement several static methods to build objects to train an HMM from
 * properties within a job configuration object.
 * 
 * @author eraldof
 * 
 */
public class TrainHmmConfig {

	/**
	 * Loging object.
	 */
	private static final Log LOG = LogFactory.getLog(TrainHmmMain.class);

	private static void addDistributedCacheFile(JobContext jobContext,
			String fileName, String linkName, String property)
			throws URISyntaxException {
		// Configuration dictionary.
		Configuration conf = jobContext.getConfiguration();

		// Add the file to the distributed cache using a link.
		LOG.info(String.format("Adding file %s in distributed cache as %s.",
				fileName, linkName));
		DistributedCache.addCacheFile(new URI(fileName + "#" + linkName), conf);
		conf.set(property, linkName);

		// Create symlinks to the cached files.
		DistributedCache.createSymlink(conf);
	}

	public static void setSmallCorpus(JobContext jobContext, String fileName)
			throws URISyntaxException {
		addDistributedCacheFile(jobContext, fileName, "smallCorpus",
				"structlearning.incorpus");
	}

	public static void setNumberOfStates(JobContext jobContext,
			int numberOfStates) {
		jobContext.getConfiguration().setInt("structlearning.numlabels",
				numberOfStates);
	}

	public static void setNumberOfSymbols(JobContext jobContext,
			int numberOfSymbols) {
		jobContext.getConfiguration().setInt("structlearning.numsymbols",
				numberOfSymbols);
	}

	public static void setNumberOfEpochs(JobContext jobContext,
			int numberOfEpochs) {
		jobContext.getConfiguration().setInt("structlearning.numepochs",
				numberOfEpochs);
	}

	public static void setLearningRate(JobContext jobContext, float learningRate) {
		jobContext.getConfiguration().setFloat("structlearning.learnrate",
				learningRate);
	}

	public static void setDefaultState(JobContext jobContext, int defaultState) {
		jobContext.getConfiguration().setInt("structlearning.defstate",
				defaultState);
	}

	public static void setSeed(JobContext jobContext, int seed) {
		jobContext.getConfiguration().setInt("structlearning.seed", seed);
	}

	public static void setLossWeight(JobContext jobContext, float lossWeight) {
		jobContext.getConfiguration().setFloat("structlearning.lossweight",
				lossWeight);
	}

	public static void setNonAnnotatedLossWeight(JobContext jobContext,
			float nonAnnotatedlossWeight) {
		jobContext.getConfiguration().setFloat(
				"structlearning.lossnonlabeledweight", nonAnnotatedlossWeight);
	}

	public static void setNonAverageWeights(JobContext jobContext,
			boolean nonAverageWeights) {
		jobContext.getConfiguration().setBoolean("structlearning.noavg",
				nonAverageWeights);
	}

	public static void setAlgorithm(JobContext jobContext, String alg) {
		jobContext.getConfiguration().set("structlearning.alg", alg);
	}

	public Hmm hmm;
	public Dataset smallDataset;
	public ViterbiInference inference;
	public StructuredAlgorithm alg;

	public static TrainHmmConfig createConfig(JobContext jobContext)
			throws ConfigException, IOException, DatasetException {

		// Configuration dictionary.
		Configuration conf = jobContext.getConfiguration();

		// Get used properties.
		String inCorpuFileName = conf.get("structlearning.incorpus");
		int numStates = conf.getInt("structlearning.numlabels", -1);
		int numSymbols = conf.getInt("structlearning.numsymbols", -1);
		int numEpochs = conf.getInt("structlearning.numepochs", 10);
		double learningRate = conf.getFloat("structlearning.learnrate", 1f);
		int defaultState = conf.getInt("structlearning.defstate", 0);
		int seed = conf.getInt("structlearning.seed", 0);
		double lossWeight = conf.getFloat("structlearning.lossweight", 1f);
		double lossNonAnnotatedWeight = conf.getFloat(
				"structlearning.lossnonlabeledweight", -1f);
		boolean averageWeights = !conf
				.getBoolean("structlearning.noavg", false);

		// Viterbi inference algorithms.
		ViterbiInference inference = new ViterbiInference(defaultState);

		// Load the small dataset.
		Dataset smallDataset = null;
		if (inCorpuFileName != null)
			smallDataset = new Dataset(inCorpuFileName, null, null);

		// Create an empty model.
		Hmm hmm = new AveragedArrayBasedHmm(numStates, numSymbols);

		// Algorithm type configuration.
		String algTypeStr = conf.get("structlearning.alg");

		StructuredAlgorithm alg = null;
		if (algTypeStr == null || algTypeStr.equals("perc")) {
			// Ordinary Perceptron implementation (Collins'): does not consider
			// customized loss functions.
			alg = new Perceptron(inference, hmm, numEpochs, learningRate, true,
					averageWeights, LearnRateUpdateStrategy.NONE);
		} else if (algTypeStr.equals("loss")) {
			// Loss-augumented implementation: considers customized loss
			// function (per-token misclassification loss).
			alg = new LossAugmentedPerceptron(inference, hmm, numEpochs,
					learningRate, lossWeight, lossNonAnnotatedWeight, 0d, true,
					averageWeights, LearnRateUpdateStrategy.NONE);
		} else if (algTypeStr.equals("afworse")) {
			// Away-from-worse implementation.
			alg = new AwayFromWorsePerceptron(inference, hmm, numEpochs,
					learningRate, lossWeight, lossNonAnnotatedWeight, 0d, true,
					averageWeights, LearnRateUpdateStrategy.NONE);
		} else if (algTypeStr.equals("tobetter")) {
			// Toward-better implementation.
			alg = new TowardBetterPerceptron(inference, hmm, numEpochs,
					learningRate, lossWeight, lossNonAnnotatedWeight, 0d, true,
					averageWeights, LearnRateUpdateStrategy.NONE);
		} else {
			throw new ConfigException("Unknown algorithm " + algTypeStr);
		}

		if (seed != 0)
			alg.setSeed(seed);

		// Store the created objects.
		TrainHmmConfig trainHmmConfig = new TrainHmmConfig();
		trainHmmConfig.alg = alg;
		trainHmmConfig.hmm = hmm;
		trainHmmConfig.inference = inference;
		trainHmmConfig.smallDataset = smallDataset;

		return trainHmmConfig;
	}

	/**
	 * Read a model saved in one or more <code>SequenceFile</code>'s.
	 * 
	 * @param numberOfStates
	 * @param numberOfSymbols
	 * @param reader
	 * @return
	 * @throws IOException
	 */
	public static Hmm loadModelFromStripes(int numberOfStates,
			int numberOfSymbols, MultipleSequenceFileReader reader)
			throws IOException {
		// Allocate an empty model with the given dimensions.
		AveragedArrayBasedHmm hmm = new AveragedArrayBasedHmm(numberOfStates,
				numberOfSymbols);

		// Read the parameter vectors in the given reader and store the values
		// in the created model.
		HmmDistributionKey key = new HmmDistributionKey();
		MapWritable map = new MapWritable();
		while (reader.next(key, map)) {
			switch (key.getType()) {

			case INITIAL_STATE_PROBABILITIES: {
				for (Entry<Writable, Writable> entry : map.entrySet()) {
					IntWritable state = (IntWritable) entry.getKey();
					DoubleWritable value = (DoubleWritable) entry.getValue();
					hmm.setInitialStateParameter(state.get(), value.get());
				}
				break;
			}

			case TRANSITION_PROBABILITIES: {
				int fromState = key.getState();
				for (Entry<Writable, Writable> entry : map.entrySet()) {
					IntWritable toState = (IntWritable) entry.getKey();
					DoubleWritable value = (DoubleWritable) entry.getValue();
					hmm.setTransitionParameter(fromState, toState.get(),
							value.get());
				}
				break;
			}

			case EMISSION_PROBABILITIES: {
				int state = key.getState();
				for (Entry<Writable, Writable> entry : map.entrySet()) {
					IntWritable symbol = (IntWritable) entry.getKey();
					DoubleWritable value = (DoubleWritable) entry.getValue();
					hmm.setEmissionParameter(state, symbol.get(), value.get());
				}
				break;
			}

			default:
				LOG.error("Unexpected probility distribution type: "
						+ key.getType().toString());
				break;

			}
		}

		return hmm;
	}

	/**
	 * Emit the model object using the stripes approach.
	 * 
	 * @param context
	 * @throws InterruptedException
	 * @throws IOException
	 */
	public void emitModelAsStripes(
			TaskInputOutputContext<?, ?, HmmDistributionKey, MapWritable> context)
			throws IOException, InterruptedException {
		if (hmm == null)
			return;

		int numStates = hmm.getNumberOfStates();

		HmmDistributionKey key = new HmmDistributionKey();
		MapWritable map = new MapWritable();

		// Initial state probabilities.
		key.setState(0);
		key.setType(DistributionType.INITIAL_STATE_PROBABILITIES);
		for (int state = 0; state < numStates; ++state) {
			double val = hmm.getInitialStateParameter(state);
			if (val != 0d)
				map.put(new IntWritable(state), new DoubleWritable(val));
		}

		if (map.size() > 0)
			context.write(key, map);

		// Transition probabilities.
		for (int fromState = 0; fromState < numStates; ++fromState) {
			// Clear key and map.
			map.clear();
			key.setState(fromState);
			key.setType(DistributionType.TRANSITION_PROBABILITIES);

			// Fill the map for the current state (fromState).
			for (int toState = 0; toState < numStates; ++toState) {
				double val = hmm.getTransitionParameter(fromState, toState);
				if (val != 0d)
					map.put(new IntWritable(toState), new DoubleWritable(val));
			}

			// Emit.
			if (map.size() > 0)
				context.write(key, map);
		}

		// Emission probabilities.
		for (int state = 0; state < numStates; ++state) {
			// Clear key and map.
			map.clear();
			key.setState(state);
			key.setType(DistributionType.EMISSION_PROBABILITIES);

			// Fill the map for the current state (fromState).
			int numSymbols = hmm.getNumberOfSymbols();
			for (int symbol = 0; symbol < numSymbols; ++symbol) {
				double val = hmm.getEmissionParameter(state, symbol);
				if (val != 0d)
					map.put(new IntWritable(symbol), new DoubleWritable(val));
			}

			// Emit.
			if (map.size() > 0)
				context.write(key, map);
		}
	}
}
