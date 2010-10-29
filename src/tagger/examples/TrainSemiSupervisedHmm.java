package tagger.examples;

import tagger.core.HmmModel;
import tagger.core.HmmTrainer;
import tagger.core.UnsupervisedHmmTrainer;
import tagger.data.Dataset;

/**
 * Semi-supervised train an HMM model using two trainsets. The first one is used
 * to supervised train an initial model. Then, an unsupervised training
 * continues from this point using the second trainset.
 * 
 * @author eraldof
 * 
 */
public class TrainSemiSupervisedHmm {

	public static void main(String[] args) throws Exception {

		if (args.length < 7) {
			System.err.print("Syntax error. Correct syntax:\n"
					+ "	<trainfile_supervised> <trainfile_unsupervised>"
					+ " <observation_feature> <state_feature> "
					+ "<modelfile> <numiterations> <smoothing>\n");
			System.exit(1);
		}

		int arg = 0;
		String trainFileNameS = args[arg++];
		String trainFileNameU = args[arg++];
		String observationFeatureLabel = args[arg++];
		String stateFeatureLabel = args[arg++];
		String modelFileName = args[arg++];
		int numIterations = Integer.parseInt(args[arg++]);
		double smoothing = Double.parseDouble(args[arg++]);

		System.out.println(String.format(
				"Unsupervised training HMM model with the following parameters:\n"
						+ "	Supervised train file: %s\n"
						+ "	Unsupervised train file: %s\n"
						+ "	Observation feature: %s\n" + "	State feature: %s\n"
						+ "	Model file: %s\n" + "	# iterations: %d\n"
						+ "	Smoothing: %f\n", trainFileNameS, trainFileNameU,
				observationFeatureLabel, stateFeatureLabel, modelFileName,
				numIterations, smoothing));

		// Load the first trainset.
		Dataset trainsetS = new Dataset(trainFileNameS);

		// Supervised train the initial model.
		HmmTrainer hmmTrainer = new HmmTrainer();
		HmmModel model = hmmTrainer.train(trainsetS, observationFeatureLabel,
				stateFeatureLabel, "0");

		// Smoothing emission probabilities.
		if (smoothing > 0.0) {
			model.setEmissionSmoothingProbability(smoothing);
			model.normalizeProbabilities();
		}

		// State labels from the initial model.
		String[] stateFeaturesV = new String[model.getNumberOfStates()];
		for (int state = 0; state < stateFeaturesV.length; ++state)
			stateFeaturesV[state] = model.getStateLabel(state);

		// Load the second trainset.
		Dataset trainsetU = new Dataset(trainFileNameU,
				trainsetS.getFeatureValueEncoding());

		// Train an HMM model.
		UnsupervisedHmmTrainer uHmmTrainer = new UnsupervisedHmmTrainer();
		uHmmTrainer.setInitialModel(model);
		model = uHmmTrainer.train(trainsetU, observationFeatureLabel,
				stateFeaturesV, numIterations);

		// Save the model.
		model.save(modelFileName);
	}
}
