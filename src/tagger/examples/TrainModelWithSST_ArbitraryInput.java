package tagger.examples;

import org.apache.log4j.Logger;

import tagger.core.SttTagger;
import tagger.data.SstTagSet;
import tagger.data.SstTagSetBIO;
import tagger.features.FeatureBuilderBasic;

/*
 * Train a new model using arbitrary input data.
 * 
 */
public class TrainModelWithSST_ArbitraryInput {

	static private Logger logger = Logger
			.getLogger(TrainModelWithSST_ArbitraryInput.class);

	public static void main(String[] args) throws Exception {

		if (args.length < 4) {
			logger.error("Syntax error: insufficient number of arguments. Correct syntax:");
			logger.error("	<train_file> <tagset_file> <number_of_epochs> <model_file>");
			System.exit(1);
		}

		int arg = 0;
		String trainFileName = args[arg++];
		String tagsetFileName = args[arg++];
		int numberOfEpochs = Integer.parseInt(args[arg++]);
		String modelFileName = args[arg++];

		logger.info("Trainning with the following parameters: ");
		logger.info("	Train file: " + trainFileName);
		logger.info("	Tagset file: " + tagsetFileName);
		logger.info("	# epochs: " + numberOfEpochs);
		logger.info("	Model file: " + modelFileName);

		FeatureBuilderBasic ftrBuilder = new FeatureBuilderBasic();
		SstTagSet tagset = new SstTagSetBIO(tagsetFileName, "UTF-8");
		SttTagger learnTagger = new SttTagger(ftrBuilder, tagset, true);
		learnTagger.train_light(
				modelFileName, 
				ftrBuilder, 
				trainFileName,
				"UTF-8", // codec of the files
				tagset, 
				false, // do not use second order HMM (use first order)
				numberOfEpochs);
	}

}
