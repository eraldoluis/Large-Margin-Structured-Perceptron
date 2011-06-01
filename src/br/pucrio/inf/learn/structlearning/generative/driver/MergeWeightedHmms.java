package br.pucrio.inf.learn.structlearning.generative.driver;

import java.util.Map.Entry;

import br.pucrio.inf.learn.structlearning.generative.core.HmmModel;


/**
 * Merge two HMMs: a usual-format and a Hadoop-format. The user also provide the
 * weight of the first model.
 * 
 * @author eraldof
 * 
 */
public class MergeWeightedHmms {

	public static void main(String[] args) throws Exception {

		if (args.length != 4) {
			System.err
					.print("Syntax error: more arguments are necessary. Correct syntax:\n"
							+ "	<model1> <model2> <weight1> <output>\n");
			System.exit(1);
		}

		int arg = 0;
		String modelFileName1 = args[arg++];
		String modelFileName2 = args[arg++];
		double weight1 = Double.parseDouble(args[arg++]);
		String outputModelFileName = args[arg++];

		System.out.println(String.format("Used arguments:\n"
				+ "\tModel file 1: %s\n" + "\tModel file 2: %s\n"
				+ "\tWeight of model 1: %f\n" + "\tOutput model file: %s\n",
				modelFileName1, modelFileName2, weight1, outputModelFileName));

		// Load the first model using the usual format.
		HmmModel model1 = new HmmModel(modelFileName1);

		// Load the second model using the Hadoop format.
		HmmModel model2 = new HmmModel(modelFileName2, false, model1);

		// Scale the first model probabilities and add to the second model.
		for (int state = 0; state < model1.getNumberOfStates(); ++state) {
			// Initial state probabilities.
			double prob1 = model1.getInitialStateProbability(state);
			model2.setInitialStateProbability(state,
					model2.getInitialStateProbability(state) + prob1 * weight1);

			// Transitions probabilities.
			for (int stateTo = 0; stateTo < model1.getNumberOfStates(); ++stateTo) {
				prob1 = model1.getTransitionProbability(state, stateTo);
				model2.setProbTransition(state, stateTo,
						model2.getTransitionProbability(state, stateTo) + prob1
								* weight1);
			}

			// Emission probabilities.
			for (Entry<Integer, Double> emission : model1.getEmissionMap(state)
					.entrySet()) {
				prob1 = emission.getValue() * weight1;
				model2.setProbEmission(state, emission.getKey(),
						emission.getValue() + prob1 * weight1);
			}
		}

		// Normalize the probabilities.
		model2.normalizeProbabilities();

		// Save the new model to the output file.
		model2.save(outputModelFileName);
	}

}
