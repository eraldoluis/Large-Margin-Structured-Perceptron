package br.pucrio.inf.learn.structlearning.generative.driver;

import br.pucrio.inf.learn.structlearning.generative.core.HmmModel;
import br.pucrio.inf.learn.structlearning.generative.core.HmmTrainer;
import br.pucrio.inf.learn.structlearning.generative.data.Dataset;
import br.pucrio.inf.learn.structlearning.generative.data.FeatureValueEncoding;

/**
 * Train an HMM model on the data within a given file and save the model in the
 * given filename. The user also provides a fixed feature-value encoding that is
 * used (all words.
 * 
 * @author eraldof
 * 
 */
public class TrainHmmWithFixedEncoding {

	public static void main(String[] args) throws Exception {

		if (args.length != 5) {
			System.err
					.print("Syntax error: more arguments are necessary. Correct syntax:\n"
							+ "	<train file>"
							+ " <observation feature>"
							+ " <state feature>"
							+ " <encoding file>"
							+ " <modelfile>" + "\n");
			System.exit(1);
		}

		int arg = 0;
		String trainFileName = args[arg++];
		String observationFeatureLabel = args[arg++];
		String stateFeatureLabel = args[arg++];
		String encodingFileName = args[arg++];
		String modelFileName = args[arg++];

		System.out.println(String.format(
				"Training HMM model with the following parameters:\n"
						+ "\tTrain file: %s\n" + "\tObservation feature: %s\n"
						+ "\tState feature: %s\n" + "\tEncoding file: %s\n"
						+ "\tModel file: %s\n", trainFileName,
				observationFeatureLabel, stateFeatureLabel, encodingFileName,
				modelFileName));

		// Load the encoding.
		FeatureValueEncoding encoding = new FeatureValueEncoding(
				encodingFileName, true);

		// Load the trainset.
		Dataset trainset = new Dataset(trainFileName, encoding);

		// Train an HMM model.
		HmmTrainer hmmTrainer = new HmmTrainer();
		HmmModel model = hmmTrainer.train(trainset, observationFeatureLabel,
				stateFeatureLabel, "0");

		// Save the model.
		model.save(modelFileName);
	}
}
