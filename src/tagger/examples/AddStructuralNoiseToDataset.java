package tagger.examples;

import java.io.IOException;
import java.util.Collection;
import java.util.LinkedList;

import tagger.data.Dataset;
import tagger.data.DatasetException;
import tagger.evaluation.Evaluation;
import tagger.evaluation.TypedChunk;
import tagger.utils.RandomGenerator;

public class AddStructuralNoiseToDataset {

	public static void main(String[] args) throws IOException, DatasetException {

		if (args.length != 5) {
			System.err
					.print("Syntax error: more arguments are necessary. Correct syntax:\n"
							+ "	<input_dataset> <output_dataset> <feature> <remove_probability> <seed>\n");
			System.exit(1);
		}

		int arg = 0;
		String inFileName = args[arg++];
		String outFileName = args[arg++];
		String featureLabel = args[arg++];
		double removeProbability = Double.parseDouble(args[arg++]);
		int seed = Integer.parseInt(args[arg++]);

		System.out.println(String.format(
				"Adding structural noise with the following parameters:\n"
						+ "\tInput file: %s\n" + "\tOutput file: %s\n"
						+ "\tEntity feature: %s\n" + "\tRemove prob: %f\n"
						+ "\tSeed: %d\n", inFileName, outFileName,
				featureLabel, removeProbability, seed));

		if (seed > 0)
			RandomGenerator.gen.setSeed(seed);

		// Load the dataset.
		Dataset dataset = new Dataset(inFileName);

		// Extract the entities.
		Evaluation ev = new Evaluation("0");
		Collection<TypedChunk> entities = ev.extractEntities(dataset,
				featureLabel);

		// Randomly remove entities.
		LinkedList<TypedChunk> keepThese = new LinkedList<TypedChunk>();
		for (TypedChunk entity : entities) {
			double prob = RandomGenerator.gen.nextDouble();
			if (prob > removeProbability)
				keepThese.add(entity);
		}

		// Tag the kept entities.
		ev.tagEntities(dataset, featureLabel, keepThese, true, false);

		dataset.save(outFileName);

		System.out.println("Original dataset has " + entities.size()
				+ " entities.");
		System.out
				.println("New dataset has " + keepThese.size() + " entities.");
	}
}
