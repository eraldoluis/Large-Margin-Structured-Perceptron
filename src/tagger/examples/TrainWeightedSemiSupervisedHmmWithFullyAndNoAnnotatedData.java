package tagger.examples;

import java.util.Vector;

import tagger.core.HmmModel;
import tagger.core.SemiSupervisedHmmTrainer;
import tagger.core.WeightedHmmTrainer;
import tagger.data.Dataset;
import tagger.data.DatasetExample;
import tagger.utils.RandomGenerator;

/**
 * Semi-supervised train an HMM model using two trainsets. The first one is used
 * as an annotated dataset and the second one is used as a non-annotated
 * dataset. Additionally, use a different weight for the annotated trainset
 * examples.
 * 
 * @author eraldof
 * 
 */
public class TrainWeightedSemiSupervisedHmmWithFullyAndNoAnnotatedData {

	public static void main(String[] args) throws Exception {

		if (args.length != 8) {
			System.err
					.print("Syntax error. Correct syntax:\n"
							+ "	<trainfile_supervised> <trainfile_nonsupervised> "
							+ "<supervised_weight> <observation_feature> <state_feature> "
							+ "<modelfile> <numiterations> <seed>\n");
			System.exit(1);
		}

		int arg = 0;
		String trainFileNameS = args[arg++];
		String trainFileNameSS = args[arg++];
		double weightS = Double.parseDouble(args[arg++]);
		String observationFeatureLabel = args[arg++];
		String stateFeatureLabel = args[arg++];
		String modelFileName = args[arg++];
		int numIterations = Integer.parseInt(args[arg++]);
		int seed = Integer.parseInt(args[arg++]);

		System.out.println(String.format(
				"Unsupervised training HMM model with the following parameters:\n"
						+ "\tSupervised train file: %s\n"
						+ "\tNon-supervised train file: %s\n"
						+ "\tSupervised weight: %f\n"
						+ "\tObservation feature: %s\n"
						+ "	State feature: %s\n" + "\tModel file: %s\n"
						+ "	# iterations: %d\n" + "\tSeed: %d\n",
				trainFileNameS, trainFileNameSS, weightS,
				observationFeatureLabel, stateFeatureLabel, modelFileName,
				numIterations, seed));

		if (seed > 0)
			RandomGenerator.gen.setSeed(seed);

		// State labels from the initial model.
		String[] stateFeaturesV = { "0", "B-PER", "I-PER", "B-LOC", "I-LOC",
				"B-ORG", "I-ORG", "B-MISC", "I-MISC" };

		// Load the first trainset (supervised).
		Dataset trainsetS = new Dataset(trainFileNameS);
		int size1 = trainsetS.getNumberOfExamples();

		// Load the second trainset (semi-supervised).
		Dataset trainsetSS = new Dataset(trainFileNameSS,
				trainsetS.getFeatureValueEncoding());
		int size2 = trainsetSS.getNumberOfExamples();

		// Join the two datasets.
		trainsetS.add(trainsetSS);
		trainsetSS = null;

		// Create and fill the weight vector.
		Vector<Object> weights = new Vector<Object>(size1 + size2);
		weights.setSize(size1 + size2);
		int idxExample = 0;
		for (; idxExample < size1; ++idxExample)
			weights.set(idxExample, weightS);
		for (; idxExample < size1 + size2; ++idxExample)
			weights.set(idxExample, 1.0);

		// Supervised train an initial model on the joined dataset.
		WeightedHmmTrainer wHmmTrainer = new WeightedHmmTrainer();
		wHmmTrainer.setWeights(weights);
		HmmModel modelSupervised = wHmmTrainer.train(trainsetS,
				observationFeatureLabel, stateFeatureLabel, "0");

		// Fill the tagged flag vector.
		Vector<Object> flags = new Vector<Object>(size1 + size2);

		// All examples from the first dataset are flagged as tagged.
		for (idxExample = 0; idxExample < size1; ++idxExample)
			flags.add(new Boolean(true));

		// All examples from the second dataset are flagged as non-tagged.
		for (; idxExample < size1 + size2; ++idxExample)
			flags.add(new Boolean(false));

		// Train an semi-supervised HMM model.
		SemiSupervisedHmmTrainer ssHmmTrainer = new SemiSupervisedHmmTrainer();
		ssHmmTrainer.setTaggedExampleFlags(flags);
		ssHmmTrainer.setWeights(weights);
		ssHmmTrainer.setInitialModel(modelSupervised);
		HmmModel model = ssHmmTrainer.train(trainsetS, observationFeatureLabel,
				stateFeatureLabel, stateFeaturesV, numIterations);

		// Save the model.
		model.save(modelFileName);
	}
}
