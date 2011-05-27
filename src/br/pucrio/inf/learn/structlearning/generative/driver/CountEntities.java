package br.pucrio.inf.learn.structlearning.generative.driver;

import java.io.IOException;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.Map.Entry;

import br.pucrio.inf.learn.structlearning.generative.data.Corpus;
import br.pucrio.inf.learn.structlearning.generative.data.DatasetExample;
import br.pucrio.inf.learn.structlearning.generative.data.DatasetException;
import br.pucrio.inf.learn.structlearning.generative.evaluation.Evaluation;
import br.pucrio.inf.learn.structlearning.generative.evaluation.TypedChunk;


public class CountEntities {

	/**
	 * @param args
	 * @throws DatasetException
	 * @throws IOException
	 */
	public static void main(String[] args) throws IOException, DatasetException {

		// Verify and parse the command-line parameters.
		if (args.length != 2) {
			System.err
					.print("Syntax error: more arguments are necessary. Correct syntax:\n"
							+ "	<dataset> <feature>\n");
			System.exit(1);
		}

		int arg = 0;
		String fileName = args[arg++];
		String featureLabel = args[arg++];

		System.out.println(String.format("Parameters:\n" + "\tDataset: %s\n"
				+ "\tFeature: %s\n", fileName, featureLabel));

		// Load the dataset.
		Corpus dataset = new Corpus(fileName);

		// Extract the entities.
		Evaluation ev = new Evaluation("0");
		HashMap<String, LinkedList<TypedChunk>> entitiesByType = new HashMap<String, LinkedList<TypedChunk>>();

		// Count objects: tokens, examples and entities.
		int numTokens = 0;
		int numExamples = dataset.getNumberOfExamples();
		int numEntities = 0;
		int numExamplesWithSomeEntity = 0;
		for (DatasetExample example : dataset) {
			int numEntitiesInThisExample = ev.extractEntitiesByType(example,
					featureLabel, entitiesByType);
			if (numEntitiesInThisExample > 0)
				++numExamplesWithSomeEntity;
			numEntities += numEntitiesInThisExample;
			numTokens += example.size();
		}

		// Calculate the mean and the standard deviation of some objects.
		double meanNumTokens = numTokens / (double) numExamples;
		double numTokensStdDev = 0.0;
		for (DatasetExample example : dataset)
			numTokensStdDev += (example.size() - meanNumTokens)
					* (example.size() - meanNumTokens);
		numTokensStdDev /= (dataset.getNumberOfExamples() - 1);
		numTokensStdDev = Math.sqrt(numTokensStdDev);

		// Print the statistics.
		System.out.println("The dataset has:");
		System.out.printf("\t%8d tokens\n", numTokens);
		System.out.printf("\t%8d examples\n", numExamples);
		System.out.printf("\t%8.2f tokens per example (stdev=%5.2f)\n",
				meanNumTokens, numTokensStdDev);
		System.out.printf("\t%8d entities\n", numEntities);
		System.out.printf("\t%8.2f entities per example\n", numEntities
				/ (double) numExamples);

		// Number of entities per type.
		for (Entry<String, LinkedList<TypedChunk>> entry : entitiesByType
				.entrySet())
			System.out.printf("\t%8d %s entities\n", entry.getValue().size(),
					entry.getKey());
	}
}
