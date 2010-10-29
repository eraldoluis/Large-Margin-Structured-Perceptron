package tagger.examples;

import org.apache.log4j.Logger;

import tagger.data.ModelDescription;
import tagger.data.SstTagSet;
import tagger.data.SstTagSetBIO;
import tagger.features.FeatureBuilderBasic;
import tagger.features.Gazetter;
import tagger.features.MorphCache;
import tagger.utils.FileDescription;

/**
 * Generate derived features (surrounding word and POS tag, word shape and
 * combined features) using the data in the input file and save the result data
 * to an output file.
 * 
 * @author eraldof
 * 
 */
public class CalculateFeaturesToFile_ArbitraryInput {

	private static Logger logger = Logger
			.getLogger(CalculateFeaturesToFile_ArbitraryInput.class);

	public static void main(String args[]) throws Exception {
		if (args.length < 3) {
			logger.error("Syntax error: insufficient number of arguments. Correct syntax:");
			logger.error("	<infile> <outfile> <tagset_file> [<gazetteerfile> <morphfile> <posmodelfile>]");
			System.exit(1);
		}

		// Parameters.
		int arg = 0;
		String inputFileName = args[arg++];
		String outputFileName = args[arg++];
		String tagsetFileName = args[arg++];

		// Optional parameters if using a gazetteer-based feature builder.
		String gazetteerFileName = null;
		String morphFileName = null;
		String posModelFileName = null;
		if (args.length > arg + 3) {
			gazetteerFileName = args[arg++];
			morphFileName = args[arg++];
			posModelFileName = args[arg++];
		}

		SstTagSet tagset = new SstTagSetBIO(tagsetFileName, "UTF-8");

		FeatureBuilderBasic ftrBuilder;
		if (gazetteerFileName == null || morphFileName == null) {
			ftrBuilder = new FeatureBuilderBasic();
		} else {
			MorphCache morphCache = new MorphCache(new FileDescription(
					morphFileName, "UTF-8", false));

			ModelDescription modelDescr = new ModelDescription(
					posModelFileName, tagset, "UTF-8", false);

			ftrBuilder = new Gazetter("", gazetteerFileName, modelDescr, 4,
					morphCache);
		}

		ftrBuilder.USE_LEMMA = false;

		logger.info("Generating features with the following parameters:" + "\n" + 
				"  Input file: " + inputFileName + "\n" + 
				"  Output file: " + outputFileName + "\n" + 
				"  Tagset file: " + tagsetFileName + "\n" + 
				"  Gazetteer file: " + gazetteerFileName + "\n" + 
				"  Morpheme file: " + morphFileName + "\n" + 
				"  POS tagger model file: " + posModelFileName);

		ftrBuilder.tagfile(inputFileName, outputFileName, "UTF-8", "UTF-8",
				true, false);
	}
}
