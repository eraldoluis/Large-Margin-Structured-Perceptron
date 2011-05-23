package br.pucrio.inf.learn.structlearning.generative.driver;

import java.io.FileInputStream;
import java.io.InputStream;
import java.io.PrintStream;
import java.util.Collection;

import br.pucrio.inf.learn.structlearning.generative.data.Dataset;
import br.pucrio.inf.learn.structlearning.generative.data.DatasetExample;
import br.pucrio.inf.learn.structlearning.generative.evaluation.Evaluation;
import br.pucrio.inf.learn.structlearning.generative.evaluation.TypedChunk;


/**
 * Convert the IOB tagging style from IOB2 to IOB1 or vice-versa. The IOB2 style
 * always uses the B tag in the beginning of an entity. Conversely, the IOB1
 * style only uses it (the B tag) when it is needed, i.e., when the previous
 * token is part of a same-type entity.
 * 
 * @author eraldof
 * 
 */
public class ConvertDatasetIobStyle {

	public static void main(String[] args) throws Exception {

		if (args.length != 4) {
			System.err
					.print("Incorrect number of arguments! The correct syntax is:\n"
							+ "	<inputfile> <outputfile> <ne_feature> ( iob1 | iob2 )\n");
			System.exit(1);
		}

		int arg = 0;
		String inFileName = args[arg++];
		String outFileName = args[arg++];
		String stateFeatureLabel = args[arg++];
		String iobStyle = args[arg++];

		if (!iobStyle.equals("iob1") && !iobStyle.equals("iob2")) {
			System.err
					.print("Incorrect number of arguments! The correct syntax is:\n"
							+ "\t<inputfile> <outputfile> <ne_feature> ( iob1 | iob2 )\n");
			System.err
			.print("\t<inputfile> may be -stdin and <outputfile> may be -stdout\n");
			System.exit(1);
		}

		if (!outFileName.equals("-stdout")) {
			System.out.printf("Arguments: \n" + "\tInput filename: %s\n"
					+ "\tOutput filename: %s\n" + "\tNE feature label: %s\n"
					+ "\tTarget IOB style: %s\n", inFileName, outFileName,
					stateFeatureLabel, iobStyle);
		}

		InputStream is = null;
		if (inFileName.equals("-stdin"))
			is = System.in;
		else
			is = new FileInputStream(inFileName);

		// Load the dataset.
		Dataset dataset = new Dataset(is);

		if (!inFileName.equals("-stdin"))
			is.close();

		// Extract the entities and re-tag them using the required style.
		Evaluation ev = new Evaluation("0");
		int feature = dataset.getFeatureIndex(stateFeatureLabel);
		for (DatasetExample example : dataset) {
			// Extract the entities from the current example.
			Collection<TypedChunk> entities = ev.extractEntities(example,
					feature);

			// Re-tag the current example.
			ev.tagEntities(example, feature, entities, true,
					iobStyle.equals("iob2"));
		}

		PrintStream ps = null;
		if (outFileName.equals("-stdout"))
			ps = System.out;
		else
			ps = new PrintStream(outFileName);

		// Save the converted dataset.
		dataset.save(ps);

		if (!outFileName.equals("-stdout"))
			ps.close();
	}
}
