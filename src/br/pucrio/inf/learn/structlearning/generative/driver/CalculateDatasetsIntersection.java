package br.pucrio.inf.learn.structlearning.generative.driver;

import java.io.IOException;
import java.util.HashSet;

import br.pucrio.inf.learn.structlearning.generative.data.Corpus;
import br.pucrio.inf.learn.structlearning.generative.data.DatasetExample;
import br.pucrio.inf.learn.structlearning.generative.data.DatasetException;


/**
 * Take two datasets and a feature within it, then calculate how many feature
 * values are in both datasets and how many are only in one of them. You may
 * also specify an additional feature and some values to filter the results. If
 * so, only tokens that has the given values are considered.
 * 
 * @author eraldof
 * 
 */
public class CalculateDatasetsIntersection {

	public static void main(String[] args) throws IOException, DatasetException {

		if (args.length < 3) {
			System.err
					.print("Arguments:\n"
							+ "	<dataset1> <dataset2> <feature_label> [<filter_feature_label> <filter_value1> <feature_value2> ...]\n");
			System.exit(1);
		}

		int arg = 0;
		String fileName1 = args[arg++];
		String fileName2 = args[arg++];
		String featureLabel = args[arg++];

		// Filter values, if provided.
		String filterFtrLabel = null;
		String[] filterFtrVals = null;
		if (args.length > arg + 1) {
			filterFtrLabel = args[arg++];
			int numFtrVals = args.length - arg;
			filterFtrVals = new String[numFtrVals];
			for (int idx = 0; arg < args.length; ++idx)
				filterFtrVals[idx] = args[arg++];
		}

		System.out.printf("Argument values:\n" + "\tDataset 1: %s\n"
				+ "\tDataset 2: %s\n" + "\tFeature label: %s\n", fileName1,
				fileName2, featureLabel);

		if (filterFtrLabel != null) {
			System.out.printf("\tFilter feature label: %s\n", filterFtrLabel);
			System.out.print("\tFilter feature values: ");
			for (String val : filterFtrVals)
				System.out.print(val + ",");
			System.out.println();
		}

		// Load the first dataset.
		Corpus dataset1 = new Corpus(fileName1);
		int ftr1 = dataset1.getFeatureIndex(featureLabel);
		int filterFtr1 = dataset1.getFeatureIndex(filterFtrLabel);
		int filterValue1 = dataset1.getFeatureValueEncoding().getCodeByLabel(
				"0");

		// Build the set of feature values in the first dataset.
		HashSet<Integer> valSet1 = new HashSet<Integer>();
		for (DatasetExample example : dataset1) {
			int len = example.size();
			for (int tkn = 0; tkn < len; ++tkn) {
				if (filterFtrLabel != null
						&& example.getFeatureValue(tkn, filterFtr1) == filterValue1)
					continue;
				valSet1.add(example.getFeatureValue(tkn, ftr1));
			}
		}

		// Load the second dataset.
		Corpus dataset2 = new Corpus(fileName2,
				dataset1.getFeatureValueEncoding());
		int ftr2 = dataset2.getFeatureIndex(featureLabel);
		// int filterFtr2 = dataset2.getFeatureIndex(filterFtrLabel);
		// int filterValue2 =
		// dataset2.getFeatureValueEncoding().getCodeByLabel("0");

		// Build the set of feature values in the second dataset.
		HashSet<Integer> valSet2 = new HashSet<Integer>();
		for (DatasetExample example : dataset2) {
			int len = example.size();
			for (int tkn = 0; tkn < len; ++tkn) {
				// TODO in the second dataset, consider all words, even when
				// filtering by entity annotation, because, for Wikipedia data
				// we do not have the true annotations.
				// if (filterFtrLabel != null
				// && example.getFeatureValue(tkn, filterFtr2) == filterValue2)
				// continue;
				valSet2.add(example.getFeatureValue(tkn, ftr2));
			}
		}

		int lenSet1 = valSet1.size();
		int lenSet2 = valSet2.size();

		// Intersection.
		valSet1.retainAll(valSet2);
		int lenInterSect = valSet1.size();
		int lenUnion = lenSet1 + lenSet2 - lenInterSect;

		System.out.printf("Results (relative to the union):\n"
				+ "\tDataset1: %d (%5.2f%%)\n" + "\tDataset2: %d (%5.2f%%)\n"
				+ "\tIntersection: %d (%5.2f%%)\n" + "\tUnion: %d (%5.2f%%)\n"
				+ "\tOnly dataset1: %d (%5.2f%%)\n"
				+ "\tOnly dataset2: %d (%5.2f%%)\n", lenSet1, lenSet1 * 100.0
				/ lenUnion, lenSet2, lenSet2 * 100.0 / lenUnion, lenInterSect,
				lenInterSect * 100.0 / lenUnion, lenUnion, lenUnion * 100.0
						/ lenUnion, lenSet1 - lenInterSect,
				(lenSet1 - lenInterSect) * 100.0 / lenUnion, lenSet2
						- lenInterSect, (lenSet2 - lenInterSect) * 100.0
						/ lenUnion);

		System.out.printf("Results (relative to the first dataset):\n"
				+ "\tCovering: %5.2f%%\n"
				+ "\tDataset2 relative size: %5.2f%%\n", lenInterSect * 100.0
				/ lenSet1, lenSet2 * 100.0 / lenSet1);
	}
}
