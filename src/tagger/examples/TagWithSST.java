package tagger.examples;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.InputStreamReader;
import java.io.PrintStream;
import java.util.Vector;

import tagger.core.SttTagger;
import tagger.data.ModelDescription;
import tagger.data.SstTagSet;
import tagger.data.SstTagSetBIO;
import tagger.features.FeatureBuilderBasic;

import org.apache.log4j.Logger;
import org.apache.log4j.BasicConfigurator;

/**
 * An example for tagging an arbitrary text.
 * 
 * Needs 'sst' property set to the path where models are.
 * 
 * @author jordi
 * 
 */
public class TagWithSST {

	private static final String ENCODING = "UTF-8";
	static Logger logger = Logger.getLogger(TagWithSST.class);

	public static void main(String args[]) throws Exception {
		BasicConfigurator.configure();

		if (args.length != 4) {
			throw new Exception(
					"four parameter needed: input-file output-file model tagset");
		}

		// path to the sst models
		String basename = System.getProperty("sst");
		if (basename == null)
			throw new Exception("property sst needed!");

		// path to opennlp components to split and tokenize the raw text
		// String OPENNLPHOME= System.getProperty("opennlp"); ;
		// if(OPENNLPHOME==null) throw new
		// Exception("property opennlp needed!");

		logger.warn("warning Start");

		// Let's call the tagger
		String inputFile = args[0];// e.g. /home/jordi/cawikiTok.txt
		String outputFile = args[1];// e.g. /home/jordi/cawiki.sst
		String modelName = args[2];// e.g.MODELS/"+"allnelemacatmodel
		String tagsetname = args[3];// e.g.benchmarks/conll/CONLL03.TAGSET

		SstTagSet tagset = new SstTagSetBIO(tagsetname, ENCODING);
		ModelDescription model = new ModelDescription(modelName, tagset,
				ENCODING, false);
		FeatureBuilderBasic fb = new FeatureBuilderBasic(model);
		SttTagger catTagger = new SttTagger(fb, model, true);

		PrintStream out;
		if (outputFile.compareTo("stdout") == 0)
			out = System.out;
		else
			out = new PrintStream(outputFile);

		try {
			BufferedReader fin;
			if (inputFile.compareTo("stdin") == 0)
				fin = new BufferedReader(new InputStreamReader(System.in));
			else
				fin = new BufferedReader(new FileReader(inputFile));

			String buff;
			boolean hasLemma = true;
			boolean hasPOS = true;
			boolean hasRes = false;

			while ((buff = fin.readLine()) != null) {
				Vector<String> tokens = new Vector<String>();
				Vector<String> res = new Vector<String>();
				Vector<String> lemma = new Vector<String>();
				Vector<String> pos = new Vector<String>();

				try {
					if (buff.startsWith("%%#D") || buff.startsWith("%%#P")) {
						out.println(buff);
					} else {
						boolean endOfSentence = (buff.length() == 0)
								|| buff.startsWith("<s>")
								|| buff.startsWith("%%#");
						while (fin.ready() && !endOfSentence) {

							int nfield = 0;
							String fields[] = buff.split("\t");

							tokens.add(fields[nfield]);
							nfield++;
							if (hasPOS) {
								if (fields.length > nfield)
									pos.add(fields[nfield]);
								else
									pos.add("NOPOS");
								nfield++;
							}
							if (hasLemma) {
								if (fields.length > nfield)
									lemma.add(fields[nfield]);
								else
									lemma.add(fields[0]);
								nfield++;
							}
							if (hasRes)
								if (fields[nfield].charAt(0) == 'O')
									res.add("0");
								else
									res.add(buff.split("\t")[1]);
							buff = fin.readLine();
							endOfSentence = (buff.length() == 0)
									|| buff.startsWith("<s>")
									|| buff.startsWith("%%#");
						}

						String[] fvpos = pos.toArray(new String[0]);
						String[] fvlemma = lemma.toArray(new String[0]);
						String[] rfinalRes = catTagger.tagSequence(
								tokens.toArray(new String[0]), fvpos, fvlemma);

						for (int rj = 0; rj < rfinalRes.length; ++rj) {
							out.print(tokens.get(rj));
							if (rfinalRes[rj].charAt(0) == '0')
								out.println("\tO");
							else
								out.println("\t" + rfinalRes[rj]);
							if (hasRes)
								out.println("\t" + res.get(rj));
						}
						out.println("");
					}
				} catch (Exception e) {
					System.err.println("Error processing line:" + buff);
					throw e;
				}
			}
		} finally {
			out.close();
		}
	}
}
/**
 * TEST TAGGING FeatureBuilderBasic fb; Gazetter rfb;
 * 
 * 
 * // Let's call the tagger String basename=
 * "/home/jordi/development/tmp/sst-light/"; try { fb = new
 * FeatureBuilderBasic(basename+"MODELS/WSJPOS_up_17",
 * basename+"DATA/WSJPOS.TAGSET.gz");
 * 
 * 
 * rfb = new Gazetter(basename,basename+"DATA/GAZ/gazlistall_minussemcor", 4,
 * basename+"MODELS/WSJc_base_gaz10_up_25", basename+"DATA/WSJ.TAGSET.gz",
 * basename+"DATA/MORPH/MORPH_CACHE");
 * 
 * 
 * String[] vs = {"The", "cat", "eats", "fish", "."};
 * 
 * // String[] vpos = {"DT", "NN", "VBZ", "NN", "."}; String[] fvpos = {};
 * 
 * SttTagger PoSTagger= new SttTagger(fb,basename+"MODELS/WSJPOS_up_17","POS");
 * SttTagger WSJTagger= new
 * SttTagger(rfb,basename+"MODELS/WSJc_base_gaz10_up_25","BIO");
 * 
 * String[] vpos = PoSTagger.tagSequence(vs, fvpos);
 * 
 * String[] rfinalRes = WSJTagger.tagSequence(vs, vpos);
 * //System.out.println("Decode Tagger Result"); for(int
 * j=0;j<rfinalRes.length;++j) {
 * System.out.println(" "+vs[j]+" "+vpos[j]+" "+rfinalRes[j]); }
 * System.out.println();
 * 
 * } catch (Exception e) { // TODO Auto-generated catch block
 * e.printStackTrace();
 **/
