package tagger.examples;

import java.util.Iterator;
import java.util.Vector;

import tagger.core.SttTagger;
import tagger.data.SstTagSet;
import tagger.data.SstTagSetBIO;
import tagger.features.FeatureBuilderBasic;
import tagger.learning.Verbose_res;

/**
 * Train an SST model on the data within a given file and evaluate the resulting
 * model on the data within another given file. Write the results in terms of
 * precision, recall and F-1; and also of the number of entities, the number of
 * predicted entities and the number of correct predicted entities.
 * 
 * @author eraldof
 * 
 */
public class TrainAndTest {

	public static void main(String[] args) throws Exception {

		if (args.length < 4) {
			System.err
					.print("Syntax error: more arguments are necessary. Correct syntax:\n"
							+ "	<trainfile> <testfile> <tagsetfile> <numepochs> [<gazetteer>]\n");
			System.exit(1);
		}

		int arg = 0;
		String trainFileName = args[arg++];
		String testFileName = args[arg++];
		String tagsetFileName = args[arg++];
		int numberOfEpochs = Integer.parseInt(args[arg++]);
		String gazetteerPath = null;

		if (arg < args.length) {
			gazetteerPath = args[arg++];
		}

		System.out
				.println("Training and evaluating with the following parameters: \n"
						+ "	Train file: "
						+ trainFileName
						+ "\n"
						+ "	Test file: "
						+ testFileName
						+ "\n"
						+ "	Tagset file: "
						+ tagsetFileName
						+ "\n"
						+ "	# epochs: " + numberOfEpochs + "\n");

		FeatureBuilderBasic fb = null;
		if (gazetteerPath == null) {
			fb = new FeatureBuilderBasic();
		} else {
		}

		SstTagSet tagset = new SstTagSetBIO(tagsetFileName, "UTF-8");
		SttTagger tagger = new SttTagger(fb, tagset, true);
		Vector<Verbose_res> tst_vr = tagger.trainAndTest(trainFileName,
				testFileName, "UTF-8", numberOfEpochs, 1.0);

		// Write precision, recall and F-1 values.
		System.out.println("SST | FeatureBasic,T=" + numberOfEpochs
				+ " |   P   |   R   |   F   |");
		Iterator<Verbose_res> it = tst_vr.iterator();
		while (it.hasNext()) {
			Verbose_res res = it.next();
			System.out.println(String.format("%s | %.2f |  %.2f |  %.2f |",
					res.L, 100 * res.getPrecision(), 100 * res.getRecall(), 100 * res.getF1()));
		}

		// Write number of entities: total, predicted and correct.
		System.out.println("SST | FeatureBasic,T=" + numberOfEpochs
				+ " | Total | Predicted | Correct |");
		it = tst_vr.iterator();
		while (it.hasNext()) {
			Verbose_res res = it.next();
			System.out.println(String.format("%s | %d |  %d |  %d |", res.L,
					res.nobjects, res.nanswers, res.nfullycorrect));
		}
	}
}
