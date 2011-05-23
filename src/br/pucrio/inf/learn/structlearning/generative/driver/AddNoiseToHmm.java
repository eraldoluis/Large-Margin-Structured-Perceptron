package br.pucrio.inf.learn.structlearning.generative.driver;

import br.pucrio.inf.learn.structlearning.generative.core.HmmModel;
import br.pucrio.inf.learn.util.RandomGenerator;

/**
 * Add a normal noise to an HMM model.
 * 
 * @author eraldof
 * 
 */
public class AddNoiseToHmm {

	public static void main(String[] args) throws Exception {

		if (args.length != 4) {
			System.err
					.print("Syntax error: more arguments are necessary. Correct syntax:\n"
							+ "	<input_model> <output_model> <standard_deviation> <seed>\n");
			System.exit(1);
		}

		int arg = 0;
		String inModelFileName = args[arg++];
		String outModelFileName = args[arg++];
		double standardDeviation = Double.parseDouble(args[arg++]);
		int seed = Integer.parseInt(args[arg++]);

		System.out.println(String.format(
				"Adding normal noise to a model with the following parameters:\n"
						+ "	Input model file: %s\n"
						+ "	Output model file: %s\n"
						+ "	Standard deviation: %f\n" + "	Seed: %d\n",
				inModelFileName, outModelFileName, standardDeviation, seed));

		// Set the seed of the random number generator.
		if (seed > 0)
			RandomGenerator.gen.setSeed(seed);

		// Load the input model.
		HmmModel model = new HmmModel(inModelFileName);

		// Add noise to the model.
		model.addNormalNoise(standardDeviation);

		// Save the noisy model.
		model.save(outModelFileName);
	}
}
