package tagger.examples;

import java.io.IOException;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.Map.Entry;

import tagger.data.Dataset;
import tagger.data.DatasetExample;
import tagger.data.DatasetException;
import tagger.data.MemoryEfficientDataset;
import tagger.evaluation.Evaluation;
import tagger.evaluation.TypedChunk;

public class CountEntitiesUsingMemoryEfficientDataset {

	/**
	 * @param args
	 * @throws DatasetException
	 * @throws IOException
	 */
	public static void main(String[] args) throws IOException, DatasetException {

		// Verify and parse the command-line parameters.
		if (args.length < 3) {
			System.err
					.print("Syntax error: more arguments are necessary. Correct syntax:\n"
							+ "	<num_ftrs> <feature_index> <dataset1>:[<dataset2>:...] [<valid_tag1>:<valid_tag2>:...]\n");
			System.exit(1);
		}

		int arg = 0;
		int numFeatures = Integer.parseInt(args[arg++]);
		int feature = Integer.parseInt(args[arg++]);

		String fileNamesStr = args[arg++];
		String[] fileNames = fileNamesStr.split(":");

		String validTagsConc = "";
		String[] validTags = null;
		if (arg < args.length) {
			validTagsConc = args[arg++];
			validTags = validTagsConc.split(":");
		}

		System.out.printf("Parameters:\n" + "\t# features: %d\n"
				+ "\tAnnotation feature index: %d\n" + "\tDataset files: %s\n"
				+ "\tValid tags: %s\n", numFeatures, feature, fileNamesStr,
				validTagsConc);
		System.out.println();

		// Load the dataset.
		Dataset dataset = new MemoryEfficientDataset(fileNames);
		for (int ftrIdx = 0; ftrIdx < numFeatures; ++ftrIdx)
			dataset.createNewFeature("ftr" + ftrIdx);

		// Extract the entities and count some objects (tokens, examples and
		// entities).
		Evaluation ev = new Evaluation("0");
		if (validTags != null && validTags.length > 0)
			ev.setValidChunkTypes(validTags);
		HashMap<String, Integer> numEntitiesByType = new HashMap<String, Integer>();
		HashMap<String, LinkedList<TypedChunk>> entitiesByType = new HashMap<String, LinkedList<TypedChunk>>();
		int numTokens = 0;
		int numExamples = 0;
		int numEntities = 0;
		int numExamplesWithSomeEntity = 0;
		for (DatasetExample example : dataset) {
			// Extract the entities within the current example.
			int numEntitiesWithinThisExample = ev.extractEntitiesByType(
					example, feature, entitiesByType);

			numEntities += numEntitiesWithinThisExample;
			if (numEntitiesWithinThisExample > 0)
				++numExamplesWithSomeEntity;
			numTokens += example.size();
			++numExamples;

			// Account the number of entities by type.
			for (String type : entitiesByType.keySet()) {
				int numEntitiesOfThisType = entitiesByType.get(type).size();
				Integer accum = numEntitiesByType.get(type);
				if (accum == null)
					accum = 0;
				accum = accum + numEntitiesOfThisType;
				numEntitiesByType.put(type, accum);

				entitiesByType.get(type).clear();
			}
		}

		// Print the statistics.
		System.out.println("The dataset has:");
		System.out.printf("\t%8d tokens\n", numTokens);
		System.out.printf("\t%8d examples\n", numExamples);

		// Calculate the mean and the standard deviation of the number of tokens
		// per example (sentence).
		double meanNumTokens = numTokens / (double) numExamples;
		double numTokensStdDev = 0.0;
		for (DatasetExample example : dataset)
			numTokensStdDev += (example.size() - meanNumTokens)
					* (example.size() - meanNumTokens);
		numTokensStdDev /= (numExamples - 1);
		numTokensStdDev = Math.sqrt(numTokensStdDev);

		// Print more statistics.
		System.out.printf("\t%8.5f tokens per example (stdev=%f)\n",
				meanNumTokens, numTokensStdDev);
		System.out.printf("\t%8d entities\n", numEntities);
		System.out.printf("\t%8.5f entities per example\n", numEntities
				/ (double) numExamples);
		System.out.printf("\t%8d examples with some entity (%.2f%%)\n",
				numExamplesWithSomeEntity, numExamplesWithSomeEntity * 100
						/ (double) numExamples);
		System.out.printf("\t%8.5f entities per example with some entity\n",
				numEntities / (double) numExamplesWithSomeEntity);

		// Number of entities per type.
		for (Entry<String, Integer> entry : numEntitiesByType.entrySet())
			System.out.printf("\t%8d %s entities (%5.2f%%)\n", entry.getValue(),
					entry.getKey(), 100 * entry.getValue()
							/ (double) numEntities);

		System.out.println();
	}
}
