package tagger.examples;

import tagger.core.HmmModel;
import tagger.data.Dataset;
import tagger.utils.RandomGenerator;

/**
 * Train an HMM model on the data within a given file and evaluate the resulting
 * model on the data within another given file. Write the results in terms of
 * precision, recall and F-1; and also the number of entities, the number of
 * predicted entities and the number of correct predicted entities.
 * 
 * @author eraldof
 * 
 */
public class GenerateDatasetFromHmmModel {

	public static void main(String[] args) throws Exception {

		if (args.length != 8) {
			System.err
					.println("Syntax error: more arguments are necessary. Correct syntax:\n"
							+ "	<modelfile> <observation_feature_label>"
							+ " <state_feature_label> <number_examples> <example_length_mean>"
							+ " <length_standard_deviation> <datasetfile> <seed>\n");
			System.exit(1);
		}

		int arg = 0;
		String modelFileName = args[arg++];
		String observationFeatureLabel = args[arg++];
		String stateFeatureLabel = args[arg++];
		int numberOfExamples = Integer.parseInt(args[arg++]);
		double exampleLengthMean = Double.parseDouble(args[arg++]);
		double exampleLengthStdDev = Double.parseDouble(args[arg++]);
		String datasetFileName = args[arg++];
		int seed = Integer.parseInt(args[arg++]);

		System.out.println(String.format(
				"Generating dataset with the following parameters:\n"
						+ "	Model file: %s\n" + "	Observation feature: %s\n"
						+ "	State feature: %s\n" + "	Number of examples: %d\n"
						+ "	Example length (mean): %f\n"
						+ "	Example length standard deviation: %f\n"
						+ "	Output filename: %s\n" + "	Seed: %d\n",
				modelFileName, observationFeatureLabel, stateFeatureLabel,
				numberOfExamples, exampleLengthMean, exampleLengthStdDev,
				datasetFileName, seed));

		// Set the seed.
		if (seed > 0)
			RandomGenerator.gen.setSeed(seed);

		// Create the output dataset.
		Dataset dataset = new Dataset();
		dataset.createNewFeature(observationFeatureLabel);
		dataset.createNewFeature(stateFeatureLabel);

		// Load the HMM model.
		HmmModel model = new HmmModel(modelFileName,
				dataset.getFeatureValueEncoding());

		// Generate examples.
		model.generateExamples(dataset, observationFeatureLabel,
				stateFeatureLabel, numberOfExamples, exampleLengthMean,
				exampleLengthStdDev);

		// Save the generated dataset.
		dataset.save(datasetFileName);
	}
}
