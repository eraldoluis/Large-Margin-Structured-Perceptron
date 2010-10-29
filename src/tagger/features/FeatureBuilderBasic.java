package tagger.features;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.PrintStream;
import java.io.Writer;
import java.util.Vector;

import org.apache.log4j.Logger;

import tagger.core.PS_HMM;
import tagger.data.DataExampleRaw;
import tagger.data.FeatureValueEncoding;
import tagger.data.ModelDescription;
import tagger.utils.Utils;

/**
 * @author jordi Given a vector of words it generates an integer vector per word
 *         representing its features
 */
public class FeatureBuilderBasic implements FeatureBuilder {
	static Logger logger = Logger.getLogger(FeatureBuilderBasic.class);

	BufferedReader in;
	protected FeatureValueEncoding FSIS;

	// public String tagsetname;//="../../../DATA/WSJ.TAGSET";
	// String morphCache;//="../../../DATA/MORPH/MORPH_CACHE";

	// Now part of model description
	// private boolean compress;
	// public String modelname;//="../../../MODELS/WSJc_base_gaz10_up_25";
	// private String encoding;

	private ModelDescription model;

	void init() throws Exception {
		// @TODO FIX initLabels();
		initFeatures();
	}

	void initFeatures() throws Exception {
		String filename = model.path + PS_HMM.FEATURE_MODEL_FILENAME_EXTENSION;
		if (model.compress)
			filename = filename + Utils.GZIP_FILENAME_EXTENSION;
		FSIS = new FeatureValueEncoding(FeatureValueEncoding.modeFeatures, filename, model.encoding);
		logger.info("***Features in the model #" + FSIS.V_STRINGS.size());
	}

	public FeatureBuilderBasic() throws Exception {
		// @TODO FIX initLabels();
		FSIS = new FeatureValueEncoding();
	}

	public FeatureBuilderBasic(ModelDescription model) throws Exception {
		// this.encoding=encoding;
		// this.compress=compress;
		// this.modelname = modelname;
		// this.tagsetname= tagsetname;
		this.model = model;
		init();
		logger.info("***Features after Gazzeters #" + FSIS.V_STRINGS.size());
	}

	// / add specific real-valued feats to standard ones
	boolean USE_R = false;

	// / add constant feature for
	boolean USE_KF = false;

	// / use word/lemma features
	boolean USE_WORDS = true;

	// / lowercase all words, generalized lowercase model
	boolean LOWERCASE = false;

	// / Use lemma attribute
	public boolean USE_LEMMA = true;

	// / USe/don't relative position features
	boolean USE_RELPOS_FEATS = false;

	// / Use morph_cache
	boolean USE_MORPH_CACHE = true;

	// / maximum length of gaz entries to use
	static final int maxspan_gaz = 10;

	// FAKE function @TODO change the callers
	public static String my_tolower(String s) {
		return s.toLowerCase();
	}

	// FAKE ?? need to modify sh!! TODO change the callers
	public static String my_tolower_sh(String s, StringBuilder sh) {

		StringBuilder ans = new StringBuilder();

		char old_char = '#';

		for (int i = 0; i < s.length(); ++i) {
			char c = s.charAt(i);
			ans.append(java.lang.Character.toLowerCase(c));

			char rg_c;
			if (java.lang.Character.isLetter(c)) // isalpha
				if (java.lang.Character.isUpperCase(c))
					rg_c = 'X';
				else
					rg_c = 'x';
			else if (java.lang.Character.isDigit(c))
				rg_c = 'd';
			else
				rg_c = c;
			if (rg_c != old_char)
				sh.append(rg_c);

			old_char = rg_c;
		}

		return ans.toString();
	}

	// @TODO fix POS parameter!!
	/*
	 * @W words
	 * 
	 * @P pos could be an empty string
	 * 
	 * @L could be an empty string
	 */
	public Vector<Vector<String>> extractFeatures(String[] W, String[] P,
			String[] L) {
		int n = W.length;
		int m = P.length;

		// @TODO jab fix parameters
		boolean lowercase = false;

		String[] LOW = new String[n];
		Vector<Vector<String>> O = new Vector<Vector<String>>();

		String[] SH = new String[n];
		String[] SB = new String[n];
		String[] SB3 = new String[n];
		String[] PR = new String[n];
		String[] PR3 = new String[n];

		// JAB
		String[] JAL = new String[n];

		for (int i = 0; i < n; ++i) {
			String w = W[i];
			if (lowercase)
				w = my_tolower(W[i]);

			String sh = "";
			String lemma = "";
			String sb = "";
			String sb3 = "";
			String pr = "";
			String pr3 = "";

			StringBuilder bsh = new StringBuilder();
			lemma = my_tolower_sh(w, bsh);
			sh = bsh.toString();

			// recheck substring
			if (lemma.length() > 2) {
				sb = lemma.substring(lemma.length() - 2, lemma.length());
				pr = lemma.substring(0, 2);
			}
			if (lemma.length() > 3) {
				sb3 = lemma.substring(lemma.length() - 3, lemma.length());
				pr3 = lemma.substring(0, 3);
			}

			// ?? JAB
			// if (pos != "")
			// P.add(pos);

			SB[i] = sb;
			SB3[i] = sb3;
			PR[i] = pr;
			PR3[i] = pr3;
			LOW[i] = lemma;
			SH[i] = sh;

			// JAB
			if (USE_LEMMA)
				JAL[i] = L[i].toLowerCase();
		}

		for (int i = 0; i < n; ++i) {
			Vector<String> W_i = new Vector<String>();
			if (USE_KF)
				W_i.add("KF");

			if (USE_RELPOS_FEATS) {
				if (i == 0)
					W_i.add("rp=begin");
				else if (i < n - 1)
					W_i.add("rp=mid");
				else
					W_i.add("rp=end");
			}

			if (i > 0) {
				if (USE_WORDS)
					W_i.add("w-1=" + LOW[i - 1]);
				if (USE_LEMMA)
					W_i.add("lm-1=" + JAL[i - 1]);

				W_i.add("sh-1=" + SH[i - 1]);
				W_i.add("sb-1=" + SB[i - 1]);
			}
			if (i < n - 1) {

				if (USE_WORDS)
					W_i.add("w+1=" + LOW[i + 1]);
				if (USE_LEMMA)
					W_i.add("lm+1=" + JAL[i + 1]);

				W_i.add("sh+1=" + SH[i + 1]);
				W_i.add("sb+1=" + SB[i + 1]);
			}
			if (USE_WORDS)
				W_i.add("w=" + LOW[i]);
			if (USE_LEMMA)
				W_i.add("lm=" + JAL[i]);

			W_i.add("sh=" + SH[i]);
			W_i.add("pr=" + PR[i]);
			W_i.add("pr3=" + PR3[i]);
			W_i.add("sb=" + SB[i]);
			W_i.add("sb3=" + SB3[i]);

			// if POS available
			if (m > 0) {
				if (i > 0) {
					W_i.add("pos-1=" + P[i - 1]);
				}
				if (i < n - 1) {
					W_i.add("pos+1=" + P[i + 1]);
				}

				W_i.add("pos=" + P[i]);
			}

			O.add(W_i);
		}

		return O;
	}

	// void extract_feats(vector<string> &W,vector<string>&
	// P,vector<vector<string> >& O,bool lowercase, vector<string>& LOW){
	// /calculates str_features
	// @DEPRECATED
	Vector<Vector<String>> OLDextractFeatures(String[] W, String[] P) {
		int n = W.length;
		int m = P.length;

		// jab fix parameters
		boolean lowercase = false;
		String[] LOW = new String[n];
		Vector<Vector<String>> O = new Vector<Vector<String>>();

		String[] SH = new String[n];
		String[] SB = new String[n];
		String[] SB3 = new String[n];
		String[] PR = new String[n];
		String[] PR3 = new String[n];

		for (int i = 0; i < n; ++i) {
			String w = W[i];
			if (lowercase)
				w = my_tolower(W[i]);

			String sh = "";
			String lemma = "";
			String sb = "";
			String sb3 = "";
			String pr = "";
			String pr3 = "";

			StringBuilder bsh = new StringBuilder();
			lemma = my_tolower_sh(w, bsh);
			sh = bsh.toString();

			// recheck substring
			if (lemma.length() > 2) {
				sb = lemma.substring(lemma.length() - 2, lemma.length());
				pr = lemma.substring(0, 2);
			}
			if (lemma.length() > 3) {
				sb3 = lemma.substring(lemma.length() - 3, lemma.length());
				pr3 = lemma.substring(0, 3);
			}

			// ?? JAB
			// if (pos != "")
			// P.add(pos);

			SB[i] = sb;
			SB3[i] = sb3;
			PR[i] = pr;
			PR3[i] = pr3;
			LOW[i] = lemma;
			SH[i] = sh;
		}

		for (int i = 0; i < n; ++i) {
			Vector<String> W_i = new Vector<String>();
			if (USE_KF)
				W_i.add("KF");

			if (USE_RELPOS_FEATS) {
				if (i == 0)
					W_i.add("rp=begin");
				else if (i < n - 1)
					W_i.add("rp=mid");
				else
					W_i.add("rp=end");
			}

			if (i > 0) {
				if (m >= 0)
					W_i.add("pos-1=" + P[i - 1]);

				if (USE_WORDS)
					W_i.add("w-1=" + LOW[i - 1]);

				W_i.add("sh-1=" + SH[i - 1]);
				W_i.add("sb-1=" + SB[i - 1]);
			}
			if (i < n - 1) {

				if (m >= 0)
					W_i.add("pos+1=" + P[i + 1]);

				if (USE_WORDS)
					W_i.add("w+1=" + LOW[i + 1]);

				W_i.add("sh+1=" + SH[i + 1]);
				W_i.add("sb+1=" + SB[i + 1]);
			}
			if (USE_WORDS)
				W_i.add("w=" + LOW[i]);

			if (m >= 0)
				W_i.add("pos=" + P[i]);

			W_i.add("sh=" + SH[i]);
			W_i.add("pr=" + PR[i]);
			W_i.add("pr3=" + PR3[i]);
			W_i.add("sb=" + SB[i]);
			W_i.add("sb3=" + SB3[i]);

			O.add(W_i);
		}
		return O;
	}

	// / encode str_features into int_features @TODO Vectro-array
	public int[][] encode(Vector<Vector<String>> O_str, boolean secondorder) {
		int _O_ = O_str.size();
		int[][] O = new int[_O_][];

		for (int i = 0; i < _O_; ++i) {
			int n = O_str.get(i).size();
			int[] O_i = new int[n];
			for (int j = 0; j < n; ++j) {
				String a = O_str.get(i).get(j);
				// @JAB?? adding only to the map (??)
				O_i[j] = (FSIS.add_update_hmap(a));

				// SecondOrder Features
				if (secondorder) {
					for (int r = j + 1; r < n; ++r) {
						String b = O_str.get(i).get(r);
						if (a.compareTo(b) > 0) // ?? a< b
							O_i[r] = (FSIS.add_update_hmap(a + "-" + b));
						else
							O_i[r] = (FSIS.add_update_hmap(b + "-" + a));
					}
				}
			}
			O[i] = (O_i);
		}
		return O;
	}

	// / encode str_features into int_features @TODO Vectro-array
	public int[][] encode(String[][] O_str, boolean secondorder) {
		int _O_ = O_str.length;
		int[][] O = new int[_O_][];
		for (int i = 0; i < _O_; ++i) {
			int n = O_str[i].length;
			int[] O_i = new int[n]; // 2TODO if second order length is
									// diferent!!!
			for (int j = 0; j < n; ++j) {
				String a = O_str[i][j];
				// @JAB?? adding only to the map (??)
				O_i[j] = (FSIS.add_update_hmap(a));

				// SecondOrder Features
				if (secondorder) {
					for (int r = j + 1; r < n; ++r) {
						String b = O_str[i][r];
						if (a.compareTo(b) > 0) // ?? a< b
							O_i[r] = (FSIS.add_update_hmap(a + "-" + b));
						else
							O_i[r] = (FSIS.add_update_hmap(b + "-" + a));
					}
				}
			}
			O[i] = O_i;
		}
		return O;
	}

	// @Override
	public Integer FSIS_update_hmap(String sid) {
		return FSIS.update_hmap(sid, false);
	}

	public Integer FSIS_update_hmap(String sid, boolean update) {
		return FSIS.update_hmap(sid, update);
	}

	public int FSIS_add_update_hmap(String w) {
		return FSIS.add_update_hmap(w);
	}

	// @Override
	public int FSIS_size() {
		// TODO Auto-generated method stub
		return FSIS.size();
	}

	public int FSIS_hsize() {
		return FSIS.h.size();
	}

	public void dump() {
		System.err.println("LSIS");
		// @TODO FIX LSIS.dump();
		System.err.println("FSIS");
		FSIS.dump();
		System.err.println("End Dump");
	}

	public void dump(Writer dlog) throws IOException {
		System.err.println("LSIS");
		// @TODO FIX LSIS.dump(dlog);
		System.err.println("FSIS");
		FSIS.dump(dlog);
		System.err.println("End Dump");
	}

	/**
	 * public static void main(String argv[]) {
	 * 
	 * FeatureBuilderBasic fb; FeatureBuilderBasic rfb; try { // Let's call the
	 * tagger String basename= "/home/jordi/development/tmp/sst-light/"; fb =
	 * new FeatureBuilderBasic(basename+"MODELS/WSJPOS_up_17",
	 * basename+"DATA/WSJPOS.TAGSET.gz", basename+"DATA/MORPH/MORPH_CACHE");
	 * 
	 * rfb = new FeatureBuilderBasic(basename+"MODELS/WSJc_base_gaz10_up_25",
	 * basename+"DATA/WSJ.TAGSET.gz", basename+"DATA/MORPH/MORPH_CACHE");
	 * 
	 * 
	 * String[] vs = {"The", "cat", "eats", "fish", "."};
	 * 
	 * String[] vpos = {"DT", "NN", "VBZ", "NN", "."}; String[] fvpos = {};
	 * 
	 * Vector<Vector<String> > res = fb.extractFeatures(vs,fvpos);
	 * 
	 * // Dump for(int i=0;i<res.size();++i) { Vector<String> vwr = res.get(i);
	 * for(int j=0;j<vwr.size();++j) System.out.print(" "+vwr.get(j));
	 * System.out.println(); }
	 * 
	 * // System.out.println("Encoding"); VectorVectorInteger resint =
	 * fb.encode(res , false);
	 * 
	 * // Dump encoding
	 * 
	 * for(int i=0;i<resint.size();++i) { VectorInteger vwr = resint.get(i);
	 * for(int j=0;j<vwr.size();++j) System.out.print(" "+vwr.get(j));
	 * System.out.println(); }
	 * 
	 * 
	 * 
	 * 
	 * //Inttagger POStagger = new Inttagger(basename+"MODELS/WSJPOS_up_17",
	 * basename+"DATA/WSJPOS.TAGSET.gz", "POS"); //VectorInteger ms =
	 * POStagger.viterbi(resint);
	 * 
	 * 
	 * ViterbiTest vt = new ViterbiTest();
	 * vt.jabLoadModel(basename+"MODELS/WSJPOS_up_17"); VectorInteger
	 * ms=vt.viterbi(resint);
	 * 
	 * 
	 * //Dump
	 * 
	 * System.out.println("Tagger Result"); for(int j=0;j<ms.size();++j)
	 * System.out.print(" "+ms.get(j)); System.out.println();
	 * 
	 * 
	 * // decode result boolean BItag=false; Vector<String> finalRes; if(BItag)
	 * finalRes= fb.checkConsistency(ms); else finalRes=fb.decode_tags(ms);
	 * 
	 * 
	 * System.out.println("Decode Tagger Result"); for(int
	 * j=0;j<finalRes.size();++j) System.out.print(" "+finalRes.get(j));
	 * System.out.println();
	 * 
	 * 
	 * //Let's try to tag Vector<String> WM = new Vector<String>(); // and
	 * easier way could be adding just the integer codification to the feature
	 * 
	 * String [] T = new String [finalRes.size()]; finalRes.toArray(T);
	 * Vector<Vector<String> > newFeatures = fb.extractFeatures(vs,T);
	 * rfb.GAZ.extract_feats(rfb.FSIS, vs, T, newFeatures, WM);
	 * 
	 * // Dump Features
	 * 
	 * for(int i=0;i<newFeatures.size();++i) { Vector<String> cvwr =
	 * newFeatures.get(i); for(int j=0;j<cvwr.size();++j)
	 * System.out.print(" "+cvwr.get(j)); System.out.println(); }
	 * 
	 * 
	 * 
	 * VectorVectorInteger rresint = rfb.encode(newFeatures , false);
	 * 
	 * 
	 * System.out.println("ENCODING"); // Dump encoding for(int
	 * i=0;i<rresint.size();++i) { VectorInteger rvwr = rresint.get(i); for(int
	 * j=0;j<rvwr.size();++j) System.out.print(" "+rvwr.get(j));
	 * System.out.println(); }
	 * 
	 * //Inttagger tagger = new
	 * Inttagger(basename+"MODELS/WSJc_base_gaz10_up_25",
	 * basename+"DATA/WSJ.TAGSET.gz", "BIO"); //VectorInteger rms =
	 * tagger.viterbi(rresint); ViterbiTest svt = new ViterbiTest();
	 * svt.jabLoadModel(basename+"MODELS/WSJc_base_gaz10_up_25"); VectorInteger
	 * rms=svt.viterbi(rresint);
	 * 
	 * //Dump
	 * 
	 * System.out.println("Tagger Result"); for(int j=0;j<rms.size();++j)
	 * System.out.print(" "+rms.get(j)); System.out.println();
	 * 
	 * // decode result BItag=true; Vector<String> rfinalRes; if(BItag)
	 * rfinalRes= rfb.checkConsistency(rms); else
	 * rfinalRes=rfb.decode_tags(rms);
	 * 
	 * 
	 * System.out.println("Decode Tagger Result"); for(int
	 * j=0;j<rfinalRes.size();++j) System.out.print(" "+rfinalRes.get(j));
	 * System.out.println();
	 * 
	 * } catch (Exception e) { e.printStackTrace(); } }
	 **/

	public String FSIS(int nid) {
		return FSIS.get(nid);
	}

	/**
	 * 
	 * @TODO ongoing refactory
	 * 
	 * @param filein
	 * @param inputEncoding
	 * @param hasPos
	 * @param hasLemma
	 * @throws IOException
	 */
	public void readMassi(String filein, String inputEncoding, boolean hasPos,
			boolean hasLemma) throws IOException {

		BufferedReader fin = Utils.getBufferedReader(filein, inputEncoding);
		String buff;
		while ((buff = fin.readLine()) != null) {
			String[] WB = buff.split("\t");
			String[] W = new String[WB.length - 1]; // first field is sentence
													// id
			DataExampleRaw data = new DataExampleRaw(WB.length - 1);

			// P is the POS vector
			String[] P = hasPos ? new String[WB.length - 1] : new String[0];
			// L lemma vector
			String[] L = hasLemma ? new String[WB.length - 1] : new String[0];

			// First element is the ID?
			for (int i = 1; i < WB.length; ++i) {
				String[] fields = WB[i].split(" ");
				W[i - 1] = fields[0];
				if (hasPos)
					P[i - 1] = fields[1];
				if (hasLemma)
					L[i - 1] = fields[2];
			}
			// String[] W = {"The", "cat", "eats", "fish"};

			int labelPos = 1;
			if (hasPos)
				++labelPos;
			if (hasPos)
				++labelPos;

			Vector<Vector<String>> features = this.extractFeatures(W, P, L);

			// write a line
			int j = 1;
			data.setId(WB[0]); // id
			for (Vector<String> feats : features) {
				data.addFeature(feats.get(0));
				for (int i = 1; i < feats.size(); ++i)
					data.addFeature(feats.get(i));
				// add lema as feat
				if (hasLemma)
					data.addFeature("lema=" + L[j]);
				// add label ? :)
				data.addLabel(W[j], P[j], L[j], WB[j].split(" ")[labelPos]);
				++j;
			}
		}
	}

	/**
	 * 
	 * Processes the examples from filein and writes their feature
	 * representation on fileout
	 * 
	 * @author jordi
	 * @param filein
	 * @param fileout
	 * @param inputEncoding
	 * @param ouputEncoding
	 * @throws IOException
	 */

	public void tagfile(String filein, String fileout, String inputEncoding,
			String outputEncoding, boolean hasPos, boolean hasLemma)
			throws IOException {
		PrintStream out = new PrintStream(fileout, outputEncoding);
		// read a line
		BufferedReader fin = new BufferedReader(new InputStreamReader(
				new FileInputStream(filein), inputEncoding));
		String buff;
		while ((buff = fin.readLine()) != null) {
			String[] WB = buff.split("\t");
			String[] W = new String[WB.length - 1]; // first field is sentence
													// id

			// P is the POS vector
			String[] P = hasPos ? new String[WB.length - 1] : new String[0];
			// L lemma vector
			String[] L = hasLemma ? new String[WB.length - 1] : new String[0];

			// First element is the ID?
			for (int i = 1; i < WB.length; ++i) {
				String[] fields = WB[i].split(" ");
				W[i - 1] = fields[0];
				if (hasPos)
					P[i - 1] = fields[1];
				if (hasLemma)
					L[i - 1] = fields[2];
			}
			// String[] W = {"The", "cat", "eats", "fish"};

			int labelPos = 1;
			if (hasLemma)
				++labelPos;
			if (hasPos)
				++labelPos;

			Vector<Vector<String>> features = this.extractFeatures(W, P, L);

			// write a line
			int j = 1;
			out.print(WB[0]); // id
			for (Vector<String> feats : features) {
				out.print("\t" + feats.get(0));
				for (int i = 1; i < feats.size(); ++i)
					out.print(" " + feats.get(i));
				// add label ? :)
				out.print(" " + WB[j].split(" ")[labelPos]);
				++j;
			}

			out.println();
		}
		out.close();
	}

}