package br.pucrio.inf.learn.structlearning.generative.driver;

import java.io.File;
import java.io.FilenameFilter;
import java.io.IOException;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.Map.Entry;

import br.pucrio.inf.learn.structlearning.generative.data.Dataset;
import br.pucrio.inf.learn.structlearning.generative.data.DatasetExample;
import br.pucrio.inf.learn.structlearning.generative.data.DatasetException;
import br.pucrio.inf.learn.structlearning.generative.evaluation.Evaluation;
import br.pucrio.inf.learn.structlearning.generative.evaluation.TypedChunk;


public class CountEntitiesUsingMemoryEfficientDataset2 {

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

		fileNames = new File(".").list(new FilenameFilter() {
			@Override
			public boolean accept(File dir, String name) {
				return name.contains("part-");
			}
		});
		fileNamesStr = fileNames.length + " files";

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

		int numTokens = 0;
		int numExamples = 0;
		int numEntities = 0;
		int numExamplesWithSomeEntity = 0;

		HashMap<String, Integer> numEntitiesByType = new HashMap<String, Integer>();
		HashMap<String, LinkedList<TypedChunk>> entitiesByType = new HashMap<String, LinkedList<TypedChunk>>();

		int idx = 0;
		for (String fileName : fileNames) {
			// Load the dataset.
			Dataset dataset = new Dataset();
			for (int ftrIdx = 0; ftrIdx < numFeatures; ++ftrIdx)
				dataset.createNewFeature("ftr" + ftrIdx);
			dataset.loadWithoutHeader(fileName);

			// Extract the entities and count some objects (tokens, examples and
			// entities).
			Evaluation ev = new Evaluation("0");
			if (validTags != null && validTags.length > 0)
				ev.setValidChunkTypes(validTags);
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

			++idx;
			if (idx % 10 == 0)
				System.out.print(".");
			if (idx % 1000 == 0)
				System.out.print(idx);
		}

		System.out.println();

		// Calculate the mean and the standard deviation of the number of
		// tokens
		// per example (sentence).
		double meanNumTokens = numTokens / (double) numExamples;

		// Print the statistics.
		System.out.println("The dataset has:");
		System.out.printf("\t%8d tokens\n", numTokens);
		System.out.printf("\t%8d examples\n", numExamples);

		// Print more statistics.
		System.out.printf("\t%8.2f tokens per example\n", meanNumTokens);
		System.out.printf("\t%8d entities\n", numEntities);
		System.out.printf("\t%8.2f entities per example\n", numEntities
				/ (double) numExamples);
		System.out.printf("\t%8d examples with some entity (%.2f%%)\n",
				numExamplesWithSomeEntity, numExamplesWithSomeEntity * 100
						/ (double) numExamples);
		System.out.printf("\t%8.2f entities per example with some entity\n",
				numEntities / (double) numExamplesWithSomeEntity);

		// Number of entities per type.
		for (Entry<String, Integer> entry : numEntitiesByType.entrySet())
			System.out.printf("\t%8d %s entities (%5.2f%%)\n",
					entry.getValue(), entry.getKey(), 100 * entry.getValue()
							/ (double) numEntities);

		System.out.println();
	}
}
