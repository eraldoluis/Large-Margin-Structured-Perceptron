package br.pucrio.inf.learn.structlearning.generative.driver;

import br.pucrio.inf.learn.structlearning.generative.core.HmmModel;
import br.pucrio.inf.learn.structlearning.generative.core.HmmModel.Smoothing;
import br.pucrio.inf.learn.structlearning.generative.data.Dataset;
import br.pucrio.inf.learn.structlearning.generative.evaluation.Evaluation;

/**
 * Evaluate a saved HMM model on the data within a given file. The tagset is
 * arbitrary. The dataset is expected to NOT contain a feature-labels header.
 * 
 * @author eraldof
 * 
 */
public class TestHmmGenericTagset {

	private static void argumentError() {
		System.err
				.print("Syntax error: more arguments are necessary. Correct syntax:\n"
						+ "	<testfile>"
						+ " <modelfile>"
						+ " <observation feature index>"
						+ " <golden feature index>"
						+ " <number of features>"
						+ " [<smoothing technique> [perstate]]\n");
		System.err
				.print("\t<smoothing>: NONE | LAPLACE | ABSOLUTE_DISCOUNTING\n");
		System.exit(1);
	}

	public static void main(String[] args) throws Exception {

		if (args.length < 5 || args.length > 7)
			argumentError();

		int arg = 0;
		String testFileName = args[arg++];
		String modelFileName = args[arg++];
		int observationFeature = Integer.parseInt(args[arg++]);
		int stateFeature = Integer.parseInt(args[arg++]);
		int numFeatures = Integer.parseInt(args[arg++]);

		// Smoothing configuration.
		Smoothing smoothing = Smoothing.ABSOLUTE_DISCOUNTING;
		boolean perStateSmoothing = false;
		if (arg < args.length) {
			smoothing = Smoothing.valueOf(args[arg++]);

			if (arg < args.length) {
				if (!args[arg++].equals("perstate"))
					argumentError();
				perStateSmoothing = true;
			}
		}

		System.out.println(String.format(
				"Evaluating HMM with the following parameters: \n"
						+ "\tTest file: %s\n" + "\tModel file: %s\n"
						+ "\tObservation feature: %d\n"
						+ "\tState feature: %d\n"
						+ "\tNumber of features: %d\n" + "\tSmoothing: %s\n"
						+ "\tPer-state smoothing: %b\n", testFileName,
				modelFileName, observationFeature, stateFeature, numFeatures,
				smoothing.toString(), perStateSmoothing));

		// Load the model.
		HmmModel model = new HmmModel(modelFileName);

		// Remove or add implicit probabilities.
		if (perStateSmoothing)
			model.removeZeroEmissionProbabilities();
		else
			model.setImplicitZeroEmissionProbabilities();

		// Create a dataset and fill the features with automatic names.
		Dataset testset = new Dataset(model.getFeatureValueEncoding());
		for (int idxFtr = 0; idxFtr < numFeatures; ++idxFtr)
			testset.createNewFeature("ftr" + idxFtr);

		// Load the dataset without a feature header.
		testset.loadWithoutHeader(testFileName);

		// Apply smoothing.
		model.applySmoothing(smoothing);

		// Test the model on a testset.
		String stateFeatureLabel = "ftr" + stateFeature;
		String observationFeatureLabel = "ftr" + observationFeature;
		String predictedFeatureLabel = stateFeatureLabel + "_predicted";
		model.tag(testset, observationFeatureLabel, predictedFeatureLabel);

		// Evaluate the predicted values.
		Evaluation ev = new Evaluation();
		double accuracy = ev.evaluateAccuracy(testset, stateFeatureLabel,
				predictedFeatureLabel);

		// Write the average accurary.
		System.out.println("Accuracy: " + accuracy);
	}
}
