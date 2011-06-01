package br.pucrio.inf.learn.structlearning.generative.driver;

import java.util.Map;

import br.pucrio.inf.learn.structlearning.generative.core.HmmModel;
import br.pucrio.inf.learn.structlearning.generative.data.Corpus;
import br.pucrio.inf.learn.structlearning.generative.evaluation.Evaluation;
import br.pucrio.inf.learn.structlearning.generative.evaluation.Performance;


/**
 * Evaluate an HMM model using a given encoding. This means that all symbols in
 * the evaluation set that are not present in the model, are treated as the
 * special __UNSEENSYMBOL__. The list of symbols is retrieved from the model
 * file and not from an encoding file.
 * 
 * 
 * @author eraldof
 * 
 */
public class TestHmmWithFixedEncoding {

	public static void main(String[] args) throws Exception {

		if (args.length != 4) {
			System.err
					.print("Syntax error: more arguments are necessary. Correct syntax:\n"
							+ "	<testfile> <modelfile> <observation_feature> <golden_state_feature>\n");
			System.exit(1);
		}

		int arg = 0;
		String testFileName = args[arg++];
		String modelFileName = args[arg++];
		String observationFeatureLabel = args[arg++];
		String stateFeatureLabel = args[arg++];

		double smooth = 1e-6;

		System.out.println(String.format(
				"Evaluating HMM with the following parameters: \n"
						+ "\tTest file: %s\n" + "\tModel file: %s\n"
						+ "\tObservation feature: %s\n"
						+ "\tState feature: %s\n" + "\tSmoothing: %e\n",
				testFileName, modelFileName, observationFeatureLabel,
				stateFeatureLabel, smooth));

		// Load the model.
		HmmModel model = new HmmModel(modelFileName);
		if (smooth > 0.0) {
			// TODO use new smoothing interface.
			// model.setEmissionSmoothingProbability(smooth);
			model.normalizeProbabilities();
			model.applyLog();
		}

		// Restrict (fix) the seen symbols to the ones present in the model.
		model.getFeatureValueEncoding().setReadOnly(true);

		// Load the testset.
		Corpus testset = new Corpus(testFileName,
				model.getFeatureValueEncoding());

		// Test the model on a testset.
		model.setUseFinalProbabilities(false);
		model.tag(testset, observationFeatureLabel, "ne");

		// Evaluate the predicted values.
		Evaluation ev = new Evaluation("0");
		Map<String, Performance> results = ev.evaluateSequences(testset,
				stateFeatureLabel, "ne");

		String[] labelOrder = { "LOC", "MISC", "ORG", "PER", "overall" };

		// Write precision, recall and F-1 values.
		System.out.println();
		System.out.println("|  *Class*  |  *P*  |  *R*  |  *F*  |");
		for (String label : labelOrder) {
			Performance res = results.get(label);
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
