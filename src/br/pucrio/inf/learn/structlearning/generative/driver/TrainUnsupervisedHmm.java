package br.pucrio.inf.learn.structlearning.generative.driver;

import br.pucrio.inf.learn.structlearning.generative.core.HmmModel;
import br.pucrio.inf.learn.structlearning.generative.core.HmmTrainer;
import br.pucrio.inf.learn.structlearning.generative.core.UnsupervisedHmmTrainer;
import br.pucrio.inf.learn.structlearning.generative.data.Corpus;

//
//
//
// TODO it is working like semi-supervised.
//
//
//


/**
 * Semi-supervised train an HMM model using two trainsets. The first one is used
 * to supervised train an initial model. Then, an unsupervised training
 * continues from this point using the second trainset.
 * 
 * @author eraldof
 * 
 */
public class TrainUnsupervisedHmm {

	public static void main(String[] args) throws Exception {

		if (args.length < 6) {
			System.err
					.print("Syntax error. Correct syntax:\n"
							+ "	<trainfile_supervised> <trainfile_unsupervised> <observation_feature> <state_feature> <modelfile> <numiterations>\n");
			System.exit(1);
		}

		int arg = 0;
		String trainFileNameS = args[arg++];
		String trainFileNameU = args[arg++];
		String observationFeatureLabel = args[arg++];
		String stateFeatureLabel = args[arg++];
		String modelFileName = args[arg++];
		int numIterations = Integer.parseInt(args[arg++]);

		System.out.println(String.format(
				"Unsupervised training HMM model with the following parameters:\n"
						+ "	Supervised train file: %s\n"
						+ "	Unsupervised train file: %s\n"
						+ "	Observation feature: %s\n" + "	State feature: %s\n"
						+ "	Model file: %s\n" + "	# iterations: %d\n",
				trainFileNameS, trainFileNameU, observationFeatureLabel,
				stateFeatureLabel, modelFileName, numIterations));

		// Load the first trainset.
		Corpus trainsetS = new Corpus(trainFileNameS);

		// Supervised train the initial model.
		HmmTrainer hmmTrainer = new HmmTrainer();
		HmmModel model = hmmTrainer.train(trainsetS, observationFeatureLabel,
				stateFeatureLabel, "0");

		// State labels.
		String[] stateFeaturesV = new String[model.getNumberOfStates()];
		for (int state = 0; state < stateFeaturesV.length; ++state)
			stateFeaturesV[state] = model.getStateLabel(state);

		// Load the second trainset.
		Corpus trainsetU = new Corpus(trainFileNameU);

		// Train an HMM model.
		UnsupervisedHmmTrainer uHmmTrainer = new UnsupervisedHmmTrainer();
		uHmmTrainer.setInitialModel(model);
		model = uHmmTrainer.train(trainsetU, observationFeatureLabel,
				stateFeaturesV, numIterations);

		// Save the model.
		model.save(modelFileName);
	}
}
