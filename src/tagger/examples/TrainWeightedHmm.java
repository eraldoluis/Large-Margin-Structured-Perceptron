package tagger.examples;

import java.io.IOException;
import java.util.Vector;

import tagger.core.HmmException;
import tagger.core.HmmModel;
import tagger.core.WeightedHmmTrainer;
import tagger.data.Dataset;
import tagger.data.DatasetException;

/**
 * Train a weighted HMM with two datasets using different weights for each
 * dataset examples.
 * 
 * @author eraldof
 * 
 */
public class TrainWeightedHmm {

	public static void main(String[] args) throws IOException,
			DatasetException, HmmException {

		if (args.length != 6) {
			System.err
					.print("Syntax error: more arguments are necessary. Correct syntax:\n"
							+ "	<trainfile1> <trainfile2> <weight1> <observation_feature> <state_feature> <modelfile>\n");
			System.exit(1);
		}

		int arg = 0;
		String trainFileName1 = args[arg++];
		String trainFileName2 = args[arg++];
		double weight1 = Double.parseDouble(args[arg++]);
		String observationFeatureLabel = args[arg++];
		String stateFeatureLabel = args[arg++];
		String modelFileName = args[arg++];

		System.out.println(String.format(
				"Training HMM model with the following parameters:\n"
						+ "	Train file 1: %s\n" + "	Train file 2: %s\n"
						+ "	Weight of trainset 1: %f\n"
						+ "	Observation feature: %s\n" + "	State feature: %s\n"
						+ "	Model file: %s\n", trainFileName1, trainFileName2,
				weight1, observationFeatureLabel, stateFeatureLabel,
				modelFileName));

		// Load the trainsets.
		Dataset trainset1 = new Dataset(trainFileName1);
		Dataset trainset2 = new Dataset(trainFileName2,
				trainset1.getFeatureValueEncoding());

		// Create the weight vector.
		Vector<Object> weights = new Vector<Object>(
				trainset1.getNumberOfExamples()
						+ trainset2.getNumberOfExamples());
		weights.setSize(trainset1.getNumberOfExamples()
				+ trainset2.getNumberOfExamples());

		int idxExample = 0;
		for (; idxExample < trainset1.getNumberOfExamples(); ++idxExample)
			weights.set(idxExample, weight1);
		for (; idxExample < weights.size(); ++idxExample)
			weights.set(idxExample, 1.0);

		// Join the two datasets.
		trainset1.add(trainset2);
		trainset2 = null;

		// Train a weighted HMM model.
		WeightedHmmTrainer hmmWeightedTrainer = new WeightedHmmTrainer();
		hmmWeightedTrainer.setWeights(weights);
		HmmModel model = hmmWeightedTrainer.train(trainset1,
				observationFeatureLabel, stateFeatureLabel, "0");

		// Save the model.
		model.save(modelFileName);
	}

}
