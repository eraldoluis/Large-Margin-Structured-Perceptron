package br.pucrio.inf.learn.structlearning.generative.driver;

import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.PrintStream;

import br.pucrio.inf.learn.structlearning.generative.data.Corpus;
import br.pucrio.inf.learn.structlearning.generative.data.DatasetException;


public class RemoveFeature {

	/**
	 * @param args
	 * @throws DatasetException
	 * @throws IOException
	 */
	public static void main(String[] args) throws IOException, DatasetException {

		// Verify and parse the command-line parameters.
		if (args.length != 3) {
			System.err.print("Arguments:\n" + "	<input> <output> <feature>\n");
			System.exit(1);
		}

		int arg = 0;
		String inFileName = args[arg++];
		String outFileName = args[arg++];
		String featureLabel = args[arg++];

		if (!outFileName.equals("-stdout")) {
			System.out.println(String.format("Parameters:\n"
					+ "\tInput filename: %s\n" + "\tOutput filename: %s\n"
					+ "\tFeature label: %s\n", inFileName, outFileName,
					featureLabel));
		}

		InputStream is = null;
		if (inFileName.equals("-stdin"))
			is = System.in;
		else
			is = new FileInputStream(inFileName);

		// Load the dataset.
		Corpus dataset = new Corpus(is);
		if (!inFileName.equals("-stdin"))
			is.close();

		// Remove the feature.
		dataset.removeFeature(featureLabel);

		PrintStream ps = null;
		if (outFileName.equals("-stdout"))
			ps = System.out;
		else
			ps = new PrintStream(outFileName);

		// Save the file.
		dataset.save(ps);
		if (!outFileName.equals("-stdout"))
			ps.close();
	}
}
