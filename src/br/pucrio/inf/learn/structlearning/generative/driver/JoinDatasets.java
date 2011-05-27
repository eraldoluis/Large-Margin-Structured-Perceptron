package br.pucrio.inf.learn.structlearning.generative.driver;

import java.io.IOException;

import br.pucrio.inf.learn.structlearning.generative.data.Corpus;
import br.pucrio.inf.learn.structlearning.generative.data.DatasetException;


public class JoinDatasets {

	public static void main(String[] args) throws IOException, DatasetException {

		if (args.length < 3) {
			System.err
					.print("Syntax error: more arguments are necessary. Correct syntax:\n"
							+ "	<output_dataset> <dataset1> <dataset2> ... <datasetn>\n");
			System.exit(1);
		}

		int arg = 0;
		String outFileName = args[arg++];

		System.out.println("Joining the datasets into " + outFileName);

		// Load the dataset.
		Corpus dataset = new Corpus(args[arg++]);
		int origSize = dataset.getNumberOfExamples();
		while (arg < args.length) {
			Corpus joinDataset = new Corpus(args[arg++], dataset.getFeatureValueEncoding());
			dataset.add(joinDataset);
		}

		dataset.save(outFileName);

		System.out.println("Original dataset has " + origSize + " sentences.");
		System.out.println("New dataset has " + dataset.getNumberOfExamples()
				+ " sentences.");
	}
}
