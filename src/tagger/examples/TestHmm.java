package tagger.examples;

import java.util.Map;

import tagger.core.HmmModel;
import tagger.data.Dataset;
import tagger.evaluation.Evaluation;
import tagger.learning.Verbose_res;

/**
 * Evaluate a saved HMM model on the data within a given file. Write the results
 * in terms of precision, recall and F-1; and also the number of entities, the
 * number of predicted entities and the number of correct predicted entities.
 * 
 * @author eraldof
 * 
 */
public class TestHmm {

	private static void argumentError() {
		System.err
				.print("Syntax error: more arguments are necessary. Correct syntax:\n"
						+ "	<testfile> <modelfile> <observation_feature> <golden_state_feature> [<smoothing probability> [perstate]]\n");
		System.exit(1);
	}

	public static void main(String[] args) throws Exception {

		if (args.length < 4)
			argumentError();

		int arg = 0;
		String testFileName = args[arg++];
		String modelFileName = args[arg++];
		String observationFeatureLabel = args[arg++];
		String stateFeatureLabel = args[arg++];

		// Smoothing configuration.
		double smoothingProbability = 1e-6;
		boolean perStateSmoothing = false;
		if (arg < args.length) {
			smoothingProbability = Double.parseDouble(args[arg++]);

			if (arg < args.length) {
				if (args[arg++].equals("perstate"))
					perStateSmoothing = true;
				else
					argumentError();
			}
		}

		System.out.println(String.format(
				"Evaluating HMM with the following parameters: \n"
						+ "\tTest file: %s\n" + "\tModel file: %s\n"
						+ "\tObservation feature: %s\n"
						+ "\tState feature: %s\n" + "\tSmoothing: %e\n"
						+ "\tPer-state smoothing: %b\n", testFileName,
				modelFileName, observationFeatureLabel, stateFeatureLabel,
				smoothingProbability, perStateSmoothing));

		// Load the model.
		HmmModel model = new HmmModel(modelFileName);
		// TODO test
		System.out.println("Min emission prob: "
				+ model.getMinimumEmissionProbability());
		System.out.println("# emissions: " + model.getNumberOfEmissions());

		if (perStateSmoothing) {
			System.out.println("Removing zero emission probs.");
			model.removeZeroEmissionProbabilities();
		} else {
			System.out.println("Setting implicit zero emission probs.");
			model.setImplicitZeroEmissionProbabilities();
		}

		System.out.println("# emissions: " + model.getNumberOfEmissions());

		// Load the testset.
		Dataset testset = new Dataset(testFileName,
				model.getFeatureValueEncoding());

		// Apply smoothing.
		if (smoothingProbability > 0.0) {
			model.setEmissionSmoothingProbability(smoothingProbability);
			model.normalizeProbabilities();
			model.applyLog();
		}

		// Test the model on a testset.
		model.setUseFinalProbabilities(false);
		model.tag(testset, observationFeatureLabel, "ne");

		// Evaluate the predicted values.
		Evaluation ev = new Evaluation("0");
		Map<String, Verbose_res> results = ev.evaluateSequences(testset,
				stateFeatureLabel, "ne");

		String[] labelOrder = { "LOC", "MISC", "ORG", "PER", "overall" };

		// Write precision, recall and F-1 values.
		System.out.println();
		System.out.println("|  *Class*  |  *P*  |  *R*  |  *F*  |");
		for (String label : labelOrder) {
			Verbose_res res = results.get(label);
			if (res == null)
				continue;
			System.out.println(String.format(
					"|  %s  |  %6.2f |  %6.2f |  %6.2f |", label,
					100 * res.getPrecision(), 100 * res.getRecall(),
					100 * res.getF1()));
		}

		// // Write number of entities: total, predicted and correct.
		// System.out.println();
		// System.out.println("^  class  ^ Total ^ Retrieved ^ Correct ^");
		// for (String label : labelOrder) {
		// Verbose_res res = results.get(label);
		// if (res == null)
		// continue;
		// System.out.println(String.format("| %7s | %5d |  %8d | %7d |",
		// label, res.nobjects, res.nanswers, res.nfullycorrect));
		// }
	}
}
