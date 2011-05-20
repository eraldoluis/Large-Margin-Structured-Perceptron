package br.pucrio.inf.learn.structlearning.hadoop.mapper;

import java.io.IOException;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.hadoop.io.MapWritable;
import org.apache.hadoop.mapreduce.Mapper;

import br.pucrio.inf.learn.mr.data.ExampleData;
import br.pucrio.inf.learn.mr.data.ExampleKey;
import br.pucrio.inf.learn.mr.data.HmmDistributionKey;
import br.pucrio.inf.learn.structlearning.hadoop.config.TrainHmmConfig;

public class TrainHmmMapper extends
		Mapper<ExampleKey, ExampleData, HmmDistributionKey, MapWritable> {

	/**
	 * Logging object.
	 */
	private static final Log LOG = LogFactory.getLog(TrainHmmMapper.class);

	/**
	 * Configuration of this training mapper.
	 */
	private TrainHmmConfig config;

	@Override
	protected void setup(Context context) throws IOException,
			InterruptedException {
		try {
			config = TrainHmmConfig.createConfig(context);
		} catch (Exception e) {
			config = null;
			LOG.error("Configuring the mapper", e);
			throw new InterruptedException(e.getMessage());
		}
	}

	@Override
	protected void map(ExampleKey key, ExampleData value, Context context)
			throws IOException, InterruptedException {
		// TODO sample from the small dataset and the additional examples (given
		// to this map).
	}

	@Override
	protected void cleanup(Context context) throws IOException,
			InterruptedException {
		// Train over the small dataset alone.
		// TODO config.alg.train(config.smallDataset.getInputs(),
		// config.smallDataset.getOutputs(), null, null);
		// Emit the resulting model.
		config.emitModelAsStripes(context);
	}

}
