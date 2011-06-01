package br.pucrio.inf.learn.structlearning.generative.driver;

import java.util.Map;

import br.pucrio.inf.learn.structlearning.generative.core.HmmModel;


/**
 * Calculate and print some statistics about an HMM model.
 * 
 * @author eraldof
 * 
 */
public class HmmModelStats {

	public static void main(String[] args) throws Exception {

		if (args.length != 1) {
			System.err
					.println("Syntax error: more arguments are necessary. Correct syntax:\n"
							+ "	<modelfile>\n");
			System.exit(1);
		}

		int arg = 0;
		String modelFileName = args[arg++];

		System.out.println(String.format(
				"Generating dataset with the following parameters:\n"
						+ "	Model file: %s\n", modelFileName));

		// Load the HMM model.
		HmmModel model = new HmmModel(modelFileName);

		int numStates = model.getNumberOfStates();
		int[] numObsPerState = new int[numStates];
		for (int state = 0; state < numStates; ++state) {
			Map<Integer, Double> emissionMap = model.getEmissionMap(state);
			numObsPerState[state] += emissionMap.size();
			System.out.println(model.getStateLabel(state) + " "
					+ numObsPerState[state]);
		}
	}
}
