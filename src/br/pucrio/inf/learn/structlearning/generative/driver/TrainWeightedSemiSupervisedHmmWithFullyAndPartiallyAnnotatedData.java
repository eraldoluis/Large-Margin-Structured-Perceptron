package br.pucrio.inf.learn.structlearning.generative.driver;

import java.util.Vector;

import br.pucrio.inf.learn.structlearning.generative.core.HmmModel;
import br.pucrio.inf.learn.structlearning.generative.core.SemiSupervisedHmmTrainer;
import br.pucrio.inf.learn.structlearning.generative.core.WeightedHmmTrainer;
import br.pucrio.inf.learn.structlearning.generative.data.Corpus;
import br.pucrio.inf.learn.structlearning.generative.data.DatasetExample;
import br.pucrio.inf.learn.util.RandomGenerator;


/**
 * Semi-supervised train an HMM model using two trainsets. The first one is used
 * as a fully-annotated dataset but the second one is used as a partially
 * annotated dataset, where 0-annotated tokens are treated as unannotated
 * examples. Additionally, use a different weight for the fully-annotated
 * trainset examples.
 * 
 * @author eraldof
 * 
 */
public class TrainWeightedSemiSupervisedHmmWithFullyAndPartiallyAnnotatedData {

	public static void main(String[] args) throws Exception {

		if (args.length != 8) {
			System.err
					.print("Syntax error. Correct syntax:\n"
							+ "	<trainfile_supervised> <trainfile_semisupervised> "
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
						+ "\tSemi-supervised train file: %s\n"
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
		Corpus trainsetS = new Corpus(trainFileNameS);
		int size1 = trainsetS.getNumberOfExamples();

		// Load the second trainset (semi-supervised).
		Corpus trainsetSS = new Corpus(trainFileNameSS,
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

		// All examples in the first dataset are flagged as tagged.
		idxExample = 0;
		for (; idxExample < size1; ++idxExample)
			flags.add(new Boolean(true));

		// The tokens tagged different of 0 are flagged as tagged. The rest are
		// flagged as untagged.
		for (; idxExample < size1 + size2; ++idxExample) {
			DatasetExample example = trainsetS.getExample(idxExample);
			Vector<Boolean> flagsEx = new Vector<Boolean>(example.size());
			for (int token = 0; token < example.size(); ++token) {
				if (example.getFeatureValueAsString(token, stateFeatureLabel)
						.equals("0"))
					flagsEx.add(false);
				else
					flagsEx.add(true);
			}

			flags.add(flagsEx);
		}

		// Train an HMM model.
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
