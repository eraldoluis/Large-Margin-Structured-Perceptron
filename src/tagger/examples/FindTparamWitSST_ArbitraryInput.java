package tagger.examples;

import org.apache.log4j.Logger;

import tagger.core.SttTagger;
import tagger.data.SstTagSet;
import tagger.data.SstTagSetBIO;
import tagger.features.FeatureBuilderBasic;

/**
 * Automatically find the best value for parameter T (number of epochs) by
 * performing a cross validation procedure.
 * 
 * @author eraldof
 * 
 */
public class FindTparamWitSST_ArbitraryInput {

	private static Logger logger = Logger
			.getLogger(FindTparamWitSST_ArbitraryInput.class);

	public static void main(String[] args) throws Exception {

		if (args.length < 5) {
			logger.error("Syntax error: insufficient number of arguments. Correct syntax:\n" +
					"	<trainfile> <testfile> <tagsetfile> <numepochs> <numcviter>\n");
			System.exit(1);
		}

		int arg = 0;
		String trainFileName = args[arg++];
		String testFileName = args[arg++];
		String tagsetFileName = args[arg++];
		int numberOfEpochs = Integer.parseInt(args[arg++]);
		int numberOfCVIterations = Integer.parseInt(args[arg++]);

		logger.info("Evaluating with the following parameters: \n" +
				"	Train file: " + trainFileName + "\n" + 
				"	Test file: " + testFileName + "\n" + 
				"	Tagset file: " + tagsetFileName + "\n" +
				"	# epochs: " + numberOfEpochs + "\n" +
				"	# CV iterations: " + numberOfCVIterations + "\n");

		FeatureBuilderBasic fb = new FeatureBuilderBasic();
		SstTagSet tagset = new SstTagSetBIO(tagsetFileName, "UTF-8");
		SttTagger tagger = new SttTagger(fb, tagset, true);
		tagger.eval_light(trainFileName, testFileName, "UTF-8", fb, tagset,
				false, numberOfEpochs, numberOfCVIterations, false, 1.0, "");
	}
}
