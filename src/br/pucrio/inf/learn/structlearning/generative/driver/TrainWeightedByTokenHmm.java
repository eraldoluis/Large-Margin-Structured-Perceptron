package br.pucrio.inf.learn.structlearning.generative.driver;

import java.io.IOException;
import java.util.Vector;

import br.pucrio.inf.learn.structlearning.generative.core.HmmException;
import br.pucrio.inf.learn.structlearning.generative.core.HmmModel;
import br.pucrio.inf.learn.structlearning.generative.core.WeightedHmmTrainer;
import br.pucrio.inf.learn.structlearning.generative.data.Corpus;
import br.pucrio.inf.learn.structlearning.generative.data.DatasetExample;
import br.pucrio.inf.learn.structlearning.generative.data.DatasetException;


/**
 * Train a weighted HMM with two datasets using different weights for each
 * dataset examples. The weight used for the first dataset is given by
 * parameter. The weight used for the second dataset depends on the token tag.
 * For tokens tagged as 0, the weight is 1.0. Otherwise, the weight is the same
 * of the first dataset.
 * 
 * @author eraldof
 * 
 */
public class TrainWeightedByTokenHmm {

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
		Corpus trainset1 = new Corpus(trainFileName1);
		Corpus trainset2 = new Corpus(trainFileName2,
				trainset1.getFeatureValueEncoding());

		int size1 = trainset1.getNumberOfExamples();
		int size2 = trainset2.getNumberOfExamples();

		// Join the two datasets.
		trainset1.add(trainset2);
		trainset2 = null;

		// Create the weight vector.
		Vector<Object> weights = new Vector<Object>(size1 + size2);
		weights.setSize(size1 + size2);

		int idxExample = 0;
		// Weight of the first dataset's examples.
		for (; idxExample < size1; ++idxExample)
			weights.set(idxExample, (Double) weight1);

		// Weight of the second dataset's examples.
		for (; idxExample < size1 + size2; ++idxExample) {
			DatasetExample example = trainset1.getExample(idxExample);

			Vector<Object> exWeights = new Vector<Object>(example.size());
			weights.set(idxExample, exWeights);

			int ftrSta = trainset1.getFeatureIndex(stateFeatureLabel);
			int ftrOutVal = trainset1.getFeatureValueEncoding().putString("0");
			for (int tkn = 0; tkn < example.size(); ++tkn) {
				if (example.getFeatureValue(tkn, ftrSta) == ftrOutVal)
					exWeights.add(1.0);
				else
					exWeights.add(weight1);
			}
		}

		// Train a weighted HMM model.
		WeightedHmmTrainer hmmWeightedTrainer = new WeightedHmmTrainer();
		hmmWeightedTrainer.setWeights(weights);
		HmmModel model = hmmWeightedTrainer.train(trainset1,
				observationFeatureLabel, stateFeatureLabel, "0");

		// Save the model.
		model.save(modelFileName);
	}
}
