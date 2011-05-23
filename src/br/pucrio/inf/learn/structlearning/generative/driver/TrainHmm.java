package br.pucrio.inf.learn.structlearning.generative.driver;

import br.pucrio.inf.learn.structlearning.generative.core.HmmModel;
import br.pucrio.inf.learn.structlearning.generative.core.HmmTrainer;
import br.pucrio.inf.learn.structlearning.generative.data.Dataset;

/**
 * Train an HMM model on the data within a given file and save the model in the
 * given filename.
 * 
 * @author eraldof
 * 
 */
public class TrainHmm {

	public static void main(String[] args) throws Exception {

		if (args.length != 4) {
			System.err
					.print("Syntax error: more arguments are necessary. Correct syntax:\n"
							+ "	<trainfile> <observation_feature> <state_feature> <modelfile>\n");
			System.exit(1);
		}

		int arg = 0;
		String trainFileName = args[arg++];
		String observationFeatureLabel = args[arg++];
		String stateFeatureLabel = args[arg++];
		String modelFileName = args[arg++];

		System.out.println(String.format(
				"Training HMM model with the following parameters:\n"
						+ "	Train file: %s\n" + "	Observation feature: %s\n"
						+ "	State feature: %s\n" + "	Model file: %s\n",
				trainFileName, observationFeatureLabel, stateFeatureLabel,
				modelFileName));

		// Load the trainset.
		Dataset trainset = new Dataset(trainFileName);

		// Train an HMM model.
		HmmTrainer hmmTrainer = new HmmTrainer();
		HmmModel model = hmmTrainer.train(trainset, observationFeatureLabel,
				stateFeatureLabel, "0");

		// Save the model.
		model.save(modelFileName);
	}
}
