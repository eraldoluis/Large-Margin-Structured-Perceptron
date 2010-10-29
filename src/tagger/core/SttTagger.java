package tagger.core;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.PrintStream;
import java.util.HashSet;
import java.util.Set;
import java.util.Vector;

import org.apache.log4j.Logger;

import tagger.data.ModelDescription;
import tagger.data.SstTagSet;
import tagger.data.Dataset;
import tagger.data.SstDataset;
import tagger.evaluation.ET;
import tagger.evaluation.Evaluation;
import tagger.features.FeatureBuilder;
import tagger.learning.LearningStatistics;
import tagger.learning.Stats;
import tagger.learning.Verbose_res;
import tagger.utils.Utils;

/**
 * @TODO A test unit is needed to verify results are comparable with the C
 *       version
 * @TODO A Performance Test will be also needed to compare against C version
 * @TODO tag sequence should be an object to avoid passing several string arrays
 * 
 * @author jordi
 * 
 */
public class SttTagger {
	private static final String LABELS_ENCODING = "UTF-8";

	static Logger logger = Logger.getLogger(SttTagger.class);

	// / BIO for BIO tagtset or POS for no BIO
	public enum TagMode {
		BIO, POS
	};

	// /Ergodic (all transition allow) or non ergodic
	boolean ergodic = true;

	// / Feature Builder
	public FeatureBuilder fb;

	// / Evaluator
	protected ET et;

	// / Hide Markov Model Avarage Perceptron
	public PS_HMM ps_hmm;

	// /
	boolean BItag;

	// /
	protected Set<String> LABELS;

	// / Model Description
	ModelDescription model;

	private SttTagger.TagMode mode;

	// / Tagset
	private String tagsetname;

	// /tagset
	public SstTagSet tagset;

	// /@TODO fix this method. it returns an instance of an ergodic or non
	// ergodic HMM
	private static PS_HMM newPSHMM(FeatureBuilder fb, SstTagSet tagset,
			boolean ergodic) {
		if (ergodic)
			return new PS_HMM(tagset);
		else {
			return new PS_HMM_NoErgodic(fb, tagset);
		}
	}

	private void minit(FeatureBuilder fb, SstTagSet tagset, boolean ergodic)
			throws Exception {
		this.ergodic = ergodic;
		this.fb = fb;
		this.tagset = tagset; // new SstTagSet(tagset,"UTF-8");
		mode = tagset.mode;
		ps_hmm = newPSHMM(fb, this.tagset, ergodic);
		et = new Evaluation();
		BItag = (mode == SttTagger.TagMode.BIO);
	}

	/**
	 * Constructor for tagging.
	 * 
	 * @param fb
	 * @param model
	 * @param ergodic
	 * @throws Exception
	 */
	public SttTagger(FeatureBuilder fb, ModelDescription model, boolean ergodic)
			throws Exception {
		minit(fb, model.tagset, ergodic);
		ps_hmm.jabLoadModel(model);
	}

	/**
	 * Constructor for learning.
	 * 
	 * @param fb2
	 * @param tagset
	 * @param ergodic
	 * @throws Exception
	 */
	public SttTagger(FeatureBuilder fb2, SstTagSet tagset, boolean ergodic)
			throws Exception {
		minit(fb2, tagset, ergodic);
		// Not sure we should always load (probably only need for training)
		loadLabels(tagset.getName());
	}

	// / DEPRECATED Constructor for learning ? HMM needs tagset
	/*
	 * public SttTagger() { et = new Evaluation(); ps_hmm = newPSHMM(); } //
	 * Constructor for learning public SttTagger(FeatureBuilderBasic fb, String
	 * tagset) throws Exception { this.fb=fb; this.tagset= new
	 * SstTagSet(tagset,LABELS_ENCODING); et = new Evaluation(); ps_hmm =
	 * newPSHMM(fb,this.tagset,ergodic); }
	 */
	/**
	 * 
	 * Tagging a sequence
	 * 
	 * @param vs
	 *            wordforms
	 * @param fvpos
	 *            pos
	 * @param fvlemma
	 *            lemmas
	 * @return sequence of tags
	 */
	public String[] tagSequence(String[] vs, String[] fvpos, String[] fvlemma) {
		return innerTagging(fb.extractFeatures(vs, fvpos, fvlemma));
	}

	/**
	 * 
	 * Tagging a sequence
	 * 
	 * @param vs
	 *            wordforms
	 * @param fvpos
	 *            pos
	 * @return sequence of tags
	 */
	public String[] tagSequence(String[] vs, String[] fvpos) {
		return innerTagging(fb.extractFeatures(vs, fvpos, new String[0]));
	}

	/**
	 * Internal call for tagging from vector of features (as strings).
	 * 
	 * @param res
	 * @return
	 */
	public String[] innerTagging(Vector<Vector<String>> res) {
		int[][] resint = fb.encode(res, false);

		// Dump encoding
		//
		// for(int i=0;i<resint.size();++i)
		// {
		// VectorInteger vwr = resint.get(i);
		// for(int j=0;j<vwr.size();++j)
		// System.out.print(" "+vwr.get(j));
		// System.out.println();
		// }

		// ADD POS
		// String basename="";
		// ?? TAGSET basename+"DATA/WSJPOS.TAGSET.gz"
		// SttTagger POStagger = new SttTagger(fb,
		// basename+"MODELS/WSJPOS_up_17", "POS");
		// int[] ms = POStagger.vt.viterbi(resint);

		int[] ms = ps_hmm.viterbi(resint);

		// Dump
		//
		// System.out.println("Tagger Result");
		// for(int j=0;j<ms.size();++j)
		// System.out.print(" "+ms.get(j));
		// System.out.println();
		//

		// decode result
		String[] finalRes;
		if (BItag)
			finalRes = tagset.checkConsistency(ms);
		else
			finalRes = tagset.decode_tags(ms);

		return finalRes;
	}

	//
	// Training evaluation methods
	//

	/**
	 * 
	 * 
	 * @param traindata
	 * @param testdata
	 * @param tagsetname
	 * @param secondorder
	 * @param T
	 * @param CV
	 * @param mode
	 * @param R
	 * @param ww
	 * @param thetafile
	 * @throws Exception
	 */
	public void eval_light(String traindata, String testdata, String encoding,
			FeatureBuilder fb, SstTagSet tagsetname, boolean secondorder,
			int T, int CV, Boolean R, Double ww, String thetafile)
			throws Exception {
		logger.info("\neval-light(" + "\n\ttraindata:" + traindata
				+ "\n\ttestdata:" + testdata + "\n\ttagsetname:" + tagsetname
				+ "\n\tsecondorder=" + secondorder + "\n\tT = " + T
				+ "\n\tCV = " + CV + "\n\tmode = " + mode + "\n\tR = " + R
				+ "\n\twordweight = " + ww + "\n\ttheta = " + thetafile + " )");
		// @DEPRECATED init("NULL",tagsetname,mode,fb);
		Vector<Vector<Integer>> G = new Vector<Vector<Integer>>();
		Vector<Vector<Integer>> G2 = new Vector<Vector<Integer>>();
		Vector<String> ID = new Vector<String>();
		Vector<String> ID2 = new Vector<String>();

		// if (!R){ // Std binary features data
		Vector<Vector<Vector<Integer>>> D = new Vector<Vector<Vector<Integer>>>();
		Vector<Vector<Vector<Integer>>> D2 = new Vector<Vector<Vector<Integer>>>();
		load_data(traindata, encoding, D, G, ID, secondorder, true);
		load_data(testdata, encoding, D2, G2, ID2, secondorder, false);

		logger.info("@JAB load ends");

		String tagsetsuff, trainsuff, testsuff;

		trainsuff = new File(traindata).getName();
		testsuff = new File(testdata).getName();
		tagsetsuff = tagset.getName();

		String description = tagsetsuff + "_" + trainsuff + "_" + testsuff
				+ ".results";

		// TODO: update by Eraldo - create the TrainingData's here to use the
		// ID's, instead of calling the other eval method.
		Dataset cvTrainData = new SstDataset(D, G, ID);
		Dataset cvTestData = new SstDataset(D2, G2, ID2);
		eval(cvTrainData, cvTestData, T, CV, description, mode, ww, thetafile);
		// eval(D,D2,G,G2,T,CV,description,mode,ww,thetafile);

		/*
		 * } else { // @JAB R EVAL
		 * 
		 * System.err.println("NOT IMPLEMENTED YES");System.exit(-1);
		 * 
		 * Zr D, D2; load_data_R(traindata,D,G,ID,secondorder,R,true);
		 * load_data_R(testdata,D2,G2,ID2,secondorder,R,true);
		 * eval_R(D,D2,G,G2,T,CV,tagsetname+"_"+T+"_"+CV,mode,ww,thetafile);
		 * 
		 * }*
		 */

	}

	/**
	 * 
	 * 
	 * @param traindata
	 * @param testdata
	 * @param tagsetname
	 * @param secondorder
	 * @param T
	 * @param CV
	 * @param mode
	 * @param R
	 * @param ww
	 * @param thetafile
	 * @throws Exception
	 */
	public void eval_light(Dataset traindata, Dataset testdata,
			FeatureBuilder fb, SstTagSet tagset, boolean secondorder, int T,
			int CV, Boolean R, Double ww, String thetafile) throws Exception {
		logger.info("\neval-light(" + "\n\ttraindata:" + traindata
				+ "\n\ttestdata:" + testdata + "\n\ttagsetname:" + tagsetname
				+ "\n\tsecondorder=" + secondorder + "\n\tT = " + T
				+ "\n\tCV = " + CV + "\n\tmode = " + mode + "\n\tR = " + R
				+ "\n\twordweight = " + ww + "\n\ttheta = " + thetafile + " )");
		// init("NULL",tagsetname,mode,fb);

		logger.info("@JAB load ends");

		String tagsetsuff, trainsuff, testsuff;

		trainsuff = new File(traindata.getName()).getName();
		testsuff = new File(testdata.getName()).getName();
		tagsetsuff = new File(tagsetname).getName();

		String description = tagsetsuff + "_" + trainsuff + "_" + testsuff
				+ ".results";
		eval(traindata, testdata, T, CV, description, mode, ww, thetafile);
		/*
		 * } else { // @JAB R EVAL
		 * 
		 * System.err.println("NOT IMPLEMENTED YES");System.exit(-1);
		 * 
		 * Zr D, D2; load_data_R(traindata,D,G,ID,secondorder,R,true);
		 * load_data_R(testdata,D2,G2,ID2,secondorder,R,true);
		 * eval_R(D,D2,G,G2,T,CV,tagsetname+"_"+T+"_"+CV,mode,ww,thetafile);
		 * 
		 * }*
		 */

	}

	protected void loadLabels(String filename) throws IOException {
		loadLabels(Utils.getBufferedReader(filename, LABELS_ENCODING));
	}

	void loadLabels(BufferedReader fin) throws IOException {
		LABELS = new HashSet<String>();
		String input;
		int k = 0;
		while ((input = fin.readLine()) != null && input.length() > 0) {
			if (input.compareTo("0") != 0 && mode != SttTagger.TagMode.POS)
				LABELS.add(input.substring(2));
			++k;
			// logger.info("Label "+LABELS.size()+" "+input);
		}
		logger.info("k set to " + k);
		this.ps_hmm.k = k;
		// @TODO JAB
		LABELS.add("ALL");
		// dump();
	}

	/*
	 * DEPRECATED @TODO use this function to initilize all the constructors
	 * private void DEPRECATEDinit(String string, String _tagsetname,
	 * SttTagger.tagmode _mode, FeatureBuilder rfb ) throws Exception {
	 * mode=_mode; tagsetname=_tagsetname;
	 * 
	 * fb = rfb; loadLabels(tagsetname); this.tagset= new
	 * SstTagSet(tagsetname,"UTF-8"); //fb.dump();
	 * 
	 * }
	 */

	/**
	 * loads data in file dname
	 * 
	 * @param dname
	 * @param D
	 * @param G
	 * @param ID
	 * @param secondorder
	 * @param R
	 * @param labeledData
	 * @throws NumberFormatException
	 * @throws IOException
	 */
	/*
	 * void load_data_R(String dname, Zr D, Vector<Vector<Integer> > G,
	 * Vector<String> ID, Boolean secondorder, Boolean R, Boolean labeledData)
	 * throws NumberFormatException, IOException{
	 * System.err.println("\ntagger_light::load_data_R("+dname+")");
	 * BufferedReader fin = new BufferedReader(new FileReader(dname));
	 * 
	 * String buff;
	 * 
	 * while((buff=fin.readLine())!=null) {
	 * 
	 * String fields[] = buff.split("[ ]");
	 * 
	 * String id = fields[0];
	 * 
	 * if (id != ""){ Vector<Integer> Labels = new Vector<Integer>();
	 * 
	 * ID.add(id);
	 * 
	 * Vector<Vector<Pair<Integer,Double> > > SentenceReal = new
	 * Vector<Vector<Pair<Integer,Double> > >();
	 * 
	 * 
	 * int tj=0; while(tj<fields.length){
	 * 
	 * Vector<String> O = new Vector<String>(); Vector<Double> O_val= new
	 * Vector<Double>();
	 * 
	 * Vector<Pair<Integer,Double> > wordFeats = new Vector<Pair<Integer,Double>
	 * >();
	 * 
	 * 
	 * 
	 * String tokens[]= fields[tj].split("[\t]"); String f = fields[0]; Double
	 * f_val = Double.parseDouble(fields[1]);
	 * 
	 * O.add(f); O_val.add(f_val);
	 * 
	 * int ti=2; while (ti<tokens.length){ f= tokens[++ti];
	 * f_val=Double.parseDouble(tokens[++ti]); O.add(f); O_val.add(f_val); } int
	 * _O_ = (labeledData) ? O.size()-1 : O.size(), i = 0; if (_O_ < 1){
	 * System.err.println("\ntoo few info in id: " + id +"\t|O| = " + _O_);
	 * System.exit(-1);}
	 * 
	 * for (; i < _O_; ++i){ Pair<Integer,Double> rf_i = new
	 * Pair<Integer,Double>
	 * (fb.FSIS_add_update_hmap(O.elementAt(i)),O_val.elementAt(i));
	 * wordFeats.add(rf_i); if (secondorder){ for (int j = i+1; j < _O_-1; ++j){
	 * Pair<Integer,Double> rf_ij= new Pair<Integer,Double>(0,0.0); if
	 * (O.elementAt(i).compareTo(O.elementAt(j))<0) //@JAB check a<b rf_ij.first
	 * = fb.FSIS_add_update_hmap(O.elementAt(i)+"-"+O.elementAt(j)); else
	 * rf_ij.first = fb.FSIS_add_update_hmap(O.elementAt(j)+"-"+O.elementAt(i));
	 * rf_ij.second = O_val.elementAt(i)*O_val.elementAt(j);
	 * wordFeats.add(rf_ij); } } } SentenceReal.add(wordFeats); if (labeledData)
	 * Labels.add(fb.FSIS_add_update_hmap(O.elementAt(i))); }
	 * D.add(SentenceReal); G.add(Labels); } }
	 * 
	 * System.err.println("\t|D| = " +D.size() +"\t|G| = " +G.size()
	 * +"\t|ID| = " +ID.size() +"\t|FIS| = " +fb.FSIS_hsize()); }
	 */

	/*
	 * FIX can not return arrays!!!
	 */
	public Dataset load_data(String dname, String encoding,
			boolean secondorder) throws IOException {

		Vector<Vector<Vector<Integer>>> ND = new Vector<Vector<Vector<Integer>>>();
		Vector<Vector<Integer>> NG = new Vector<Vector<Integer>>();
		Vector<String> NID = new Vector<String>();
		// call
		load_data(dname, encoding, ND, NG, NID, secondorder, true);

		Dataset td = new SstDataset(ND, NG, NID);
		td.setName(dname);
		return td;
	}

	/**
	 * loads data from dname
	 * 
	 * @param dname
	 * @param D
	 * @param G
	 * @param ID
	 * @param secondorder
	 * @throws IOException
	 */
	void load_data(String dname, String encoding,
			Vector<Vector<Vector<Integer>>> D, Vector<Vector<Integer>> G,
			Vector<String> ID, boolean secondorder, boolean updateFeatureBuilder)
			throws IOException {
		logger.info("\ntagger_light::load_data(" + dname + ")");

		BufferedReader fin = Utils.getBufferedReader(dname, encoding);

		String buff;

		while ((buff = fin.readLine()) != null) {
			// System.err.println("read line "+buff);
			processLine(buff, D, G, ID, secondorder, updateFeatureBuilder);
		}
		// TRACE System.err.println("TRACE:"+dname);
		// TRACE
		// TRACE FileWriter dlog = new FileWriter(dname+"loadmodel.log");

		// D
		// TRACE for(int i=0;i<D.size();++i)
		// TRACE for(int j=0;j<D.get(i).size();++j)
		// TRACE for(int k=0;k<D.get(i).get(j).size();++k)
		// TRACE dlog.write(i+" "+j+" "+k+" "+(D.get(i).get(j).get(k))+"\n");
		// TRACE dlog.write("\n");
		// G
		// TRACE for(int i=0;i<G.size();++i)
		// TRACE for(int j=0;j<G.get(i).size();++j)
		// TRACE dlog.write(i+" "+j+" "+(G.get(i).get(j))+"\n");
		// TRACE dlog.write("\n");

		// TRACE FileWriter ldlog = new FileWriter(dname+"LSISloadmodel.log");
		// TRACE fb.LSIS.dump(ldlog);
		// TRACE ldlog.close();

		// TRACE FileWriter fdlog = new FileWriter(dname+"FSISloadmodel.log");
		// TRACE fb.FSIS.dump(fdlog);
		// TRACE fdlog.close();

		/**
		 * LSIS for (HM::const_iterator p = LSI.begin( ); p != LSI.end( ); ++p)
		 * dlog.write( (p->first)+":"+(p->second)+ "\n"); dlog.write("\n");
		 * 
		 * for(int i=0;i<FIS.size();++i) dlog.write(i+" "+(FIS[i])+"\n");
		 * dlog.write("\n");
		 * 
		 * 
		 * // FSIS for (HM::const_iterator p = LSI.begin( ); p != LSI.end( );
		 * ++p) dlog.write( (p->first)+":"+(p->second)+ "\n"); dlog.write("\n");
		 * 
		 * for(int i=0;i<LIS.size();++i) dlog.write(i+" "+(LIS[i])+"\n");
		 * dlog.write("\n");
		 **/
		// TRACE dlog.close();

		logger.info("\t|D| = " + D.size() + "\t|G| = " + G.size() + "\t|ID| = "
				+ ID.size() + "\t|FIS| = " + fb.FSIS_hsize());

	}

	// /**
	// * loads data from dname
	// *
	// * @param dname
	// * @param secondorder
	// * @throws IOException
	// */
	// void load_data(String dname, String encoding, TrainingData training,
	// TrainingData dev, boolean secondorder) throws IOException {
	// logger.info("\ntagger_light::load_data(" + dname + ")");
	//
	// BufferedReader fin = Utils.getBufferedReader(dname, encoding);
	//
	// String buff;
	// TrainingData current;
	// while ((buff = fin.readLine()) != null) {
	// if (true)
	// current = training;
	// else {
	// current = dev;
	// ;
	// current.processLine(buff, secondorder);
	// }
	// }
	// }

	/**
	 * 
	 * version of reading TrainingData using vectors
	 * 
	 * @param buff
	 * @param D
	 * @param G
	 * @param ID
	 * @param secondorder
	 */
	public void processLine(String buff, Vector<Vector<Vector<Integer>>> D,
			Vector<Vector<Integer>> G, Vector<String> ID, boolean secondorder,
			boolean updateFeatureBuilder) {
		String fields[] = buff.split("[\t]");

		String id = fields[0];

		if (id != "") {
			Vector<Integer> L = new Vector<Integer>();
			Vector<Vector<Integer>> S = new Vector<Vector<Integer>>();
			ID.add(id);

			for (int tj = 1; tj < fields.length; tj++) {
				Vector<String> O = new Vector<String>();
				Vector<Integer> O_int = new Vector<Integer>();

				String tokens[] = fields[tj].split("[ ]");
				// String f = fields[0];
				// O.add(f);
				// System.err.println("READ:"+fields[tj]);
				int ti = 0;
				while (ti < tokens.length) {
					String f = tokens[ti++];
					O.add(f);
				}

				int _O_ = O.size();
				if (_O_ < 2) {
					logger.error("\ntoo few info in id: " + id);
				}
				int i;
				for (i = 0; i < _O_ - 1; ++i) {
					String a = O.elementAt(i);

					Integer v = fb.FSIS_update_hmap(a, updateFeatureBuilder);
					if (v != null)
						O_int.add(v);
					else {
						// logger.info("Feature " + a + " not encoded");
					}

					// @JAB todo looks like reimplemented in FeatureBuilder
					if (secondorder) {
						for (int j = i + 1; j < _O_ - 1; ++j) {
							String b = O.elementAt(j);

							// JAB CHECK is the same in C++ a<b
							if (a.compareTo(b) < 0) {
								v = fb.FSIS_update_hmap(a + "-" + b,
										updateFeatureBuilder);
								if (v != null)
									O_int.add(v);
								else {
									logger.info("Feature " + a + "-" + b
											+ " not encoded");
								}

							} else {
								v = fb.FSIS_update_hmap(b + "-" + a,
										updateFeatureBuilder);
								if (v != null)
									O_int.add(v);
								else {
									logger.info("Feature " + b + "-" + a
											+ " not encoded");
								}

							}
						}
					}
				}
				S.add(O_int);

				// @JAB check Adding the last label (Training) if we use FSIS
				// will not be in the right order
				L.add(tagset.LSIS_add_update_hmap(O.elementAt(i)));
			}
			D.add(S);
			G.add(L);
		}
	}

	/**
	 * Evaluate the SttTagger using the training dataset (D,G) and the testing
	 * dataset (D2,G2).
	 * 
	 * The parameter T specifies the number of epochs, i.e., how many times the
	 * algorithm will be trainned over the whole training dataset. After each
	 * epoch, the current model is evaluated on the testing dataset.
	 * 
	 * @param D
	 *            feature values of the training dataset, i.e., word/token
	 *            sequences.
	 * @param D2
	 *            feature values of the testing dataset.
	 * @param G
	 *            correct/golden label sequences of the training dataset.
	 * @param G2
	 *            correct label sequences of the testing dataset.
	 * @param T
	 *            the number of epochs.
	 * @param CV
	 *            the
	 * @param descr
	 * @param mode
	 * @param ww
	 * @param thetafile
	 * @throws Exception
	 */
	public void eval(Vector<Vector<Vector<Integer>>> D,
			Vector<Vector<Vector<Integer>>> D2, Vector<Vector<Integer>> G,
			Vector<Vector<Integer>> G2, int T, int CV, String descr,
			SttTagger.TagMode mode, Double ww, String thetafile)
			throws Exception {
		Dataset cvTrainData = new SstDataset(D, G);
		Dataset cvTestData = new SstDataset(D2, G2);
		eval(cvTrainData, cvTestData, T, CV, descr, mode, ww, thetafile);
	}

	// //////////////////////////////////EVAL
	/*
	 * public void eval(Vector<Vector<Vector<Integer> > > D,
	 * Vector<Vector<Vector<Integer>>> D2, Vector<Vector<Integer> > G,
	 * Vector<Vector<Integer> > G2, int T, int CV, String descr, String mode,
	 * Double ww, String thetafile) throws Exception{ boolean no_special_symbol
	 * = false;
	 * 
	 * //Evaluation object LearningStatistics res = new LearningStatistics(); //
	 * Initialization for evaluation
	 * 
	 * res.init(CV,T,LABELS); Vector<Vector<Double> > CV_T_ALL = new
	 * Vector<Vector<Double> >(); Vector<Vector<Vector<Double> > > CV_POS_RES =
	 * new Vector<Vector<Vector<Double> > >();
	 * 
	 * TrainingData cvTrainData= new TrainingData(D,G); TrainingData cvTestData=
	 * new TrainingData(D2,G2);
	 * 
	 * //Crossvalidation loop for (int cv = 0; cv < CV; ++cv){
	 * System.err.print("\nCrosValidation cv = " + cv);
	 * 
	 * Vector<Double> T_ALL = new Vector<Double>(); Vector<Vector<Double> >
	 * POS_RES= new Vector<Vector<Double> >();
	 * 
	 * PS_HMM ps_hmm_cv= newPSHMM();
	 * ps_hmm_cv.init(fb.LSIS_size(),fb,no_special_symbol, ww);
	 * 
	 * 
	 * for (int t = 0; t < T; ++t){ System.err.println("Iteration T "+t);
	 * ps_hmm_cv.train(cvTrainData.D,cvTrainData.G); //D1, G1 int[][] tst_guess
	 * = ps_hmm_cv.guess_sequences(cvTestData.D); //DA" double tst_f = 0; //dump
	 * sequence //TRACE Utils.arraydump(D2,"inputtst_guess", "inputguess.dlog");
	 * //TRACE Utils.arraydump(DA2,"inputtst_guess", "realinputguess.dlog");
	 * //TRACE Utils.arraydump(tst_guess,"tst_guess", "guess.dlog");
	 * 
	 * if (mode == "POS"){ // ERROR possible itemized_res is return
	 * Vector<Double> itemized_res =new Vector<Double>(); //@TODO FIX size
	 * problem fb.LSIS_size() LABELS.size() tst_f =
	 * ET.evaluate_pos(cvTestData.G,tst_guess,fb,LABELS.size(),itemized_res);
	 * //GA2 POS_RES.add(itemized_res); T_ALL.add(tst_f); } else {
	 * 
	 * //@TODO PROBLEM tst_vr returned Vector<Verbose_res> tst_vr = new
	 * Vector<Verbose_res>(); //@TODO FIX size problem fb.LSIS_size()
	 * LABELS.size() tst_f = ET.evaluate_sequences(cvTestData.G,tst_guess,fb,
	 * LABELS.size(),tst_vr); // GA"
	 * 
	 * res.update(cv,t,tst_f,tst_vr);
	 * 
	 * } System.err.print("\ttst = " + tst_f); } CV_T_ALL.add(T_ALL);
	 * CV_POS_RES.add(POS_RES); }
	 * 
	 * if (mode != "POS") res.print_res(descr+".eval",CV,T); else {
	 * 
	 * FileOutputStream fos = new FileOutputStream(descr+".eval_pos");
	 * PrintStream out = new PrintStream(fos); for (int j = 0; j < T; ++j){
	 * Vector<Double> all_j = new Vector<Double>(); for (int i = 0; i < CV; ++i)
	 * all_j.add(CV_T_ALL.get(i).get(j)); double mu_all_j = Stats.mean(all_j);
	 * double err_all_j = Stats.std_err(all_j,mu_all_j); out.println( mu_all_j +
	 * "\t" + err_all_j); }
	 * 
	 * 
	 * out .println();out .println();
	 * 
	 * for (int r = 0; r <fb.LSIS_size(); ++r) out.print(fb.LSIS(r) + " "); out
	 * .println();
	 * 
	 * for (int j1 = 0; j1 < T; ++j1){ for (int r = 0; r<fb.LSIS_size(); ++r){
	 * Vector<Double> pos_tj = new Vector<Double>(); for (int i = 0; i < CV;
	 * ++i) pos_tj.add(CV_POS_RES.get(i).get(j1).get(r)); double mu_tj =
	 * Stats.mean(pos_tj); out.print(mu_tj + " "); } out .println(); }
	 * out.close(); }
	 * 
	 * }
	 */

	public void eval(Dataset cvTrainData, Dataset cvTestData, int T,
			int CV, String descr, SttTagger.TagMode mode, Double ww,
			String thetafile) throws Exception {
		boolean no_special_symbol = false;

		// Evaluation object
		LearningStatistics stats = new LearningStatistics();
		// Initialization for evaluation

		stats.init(CV, T, LABELS);
		Vector<Vector<Double>> CV_T_ALL = new Vector<Vector<Double>>();
		Vector<Vector<Vector<Double>>> CV_POS_RES = new Vector<Vector<Vector<Double>>>();

		// Crossvalidation loop
		for (int cv = 0; cv < CV; ++cv) {
			System.err.print("\nCrosValidation cv = " + cv);

			Vector<Double> T_ALL = new Vector<Double>();
			Vector<Vector<Double>> POS_RES = new Vector<Vector<Double>>();

			PS_HMM ps_hmm_cv = newPSHMM(fb, tagset, ergodic);
			ps_hmm_cv.init(tagset.LSIS_size(), fb, no_special_symbol, ww);

			for (int t = 0; t < T; ++t) {
				System.err.println("Iteration T " + t);
				ps_hmm_cv.train(cvTrainData);// (cvTrainData.D,cvTrainData.G);
												// //D1, G1
				int[][] tst_guess = ps_hmm_cv.guess_sequences(cvTestData);// (cvTestData.D);
																			// //DA"
				double tst_f = 0;
				// dump sequence
				// TRACE Utils.arraydump(D2,"inputtst_guess",
				// "inputguess.dlog");
				// TRACE Utils.arraydump(DA2,"inputtst_guess",
				// "realinputguess.dlog");
				// TRACE Utils.arraydump(tst_guess,"tst_guess", "guess.dlog");

				if (mode == SttTagger.TagMode.POS) {
					// ERROR possible itemized_res is return
					Vector<Double> itemized_res = new Vector<Double>();
					// @TODO FIX size problem fb.LSIS_size() LABELS.size()
					// tst_f =
					// ET.evaluate_pos(cvTestData.G,tst_guess,fb,LABELS.size(),itemized_res);
					// //GA2
					tst_f = et.evaluate_pos(cvTestData, tst_guess, tagset,
							LABELS.size(), itemized_res); // GA2
					POS_RES.add(itemized_res);
					T_ALL.add(tst_f);
				} else {

					// @TODO PROBLEM tst_vr returned
					Vector<Verbose_res> tst_vr = new Vector<Verbose_res>();
					// @TODO FIX size problem fb.LSIS_size() LABELS.size()
					// tst_f = ET.evaluate_sequences(cvTestData.G,tst_guess,fb,
					// LABELS.size(),tst_vr); // GA"
					tst_f = et.evaluate_sequences(cvTestData, tst_guess,
							tagset, LABELS.size(), tst_vr); // GA"

					stats.update(cv, t, tst_f, tst_vr);

				}
				System.err.print("\ttst = " + tst_f);
			}
			CV_T_ALL.add(T_ALL);
			CV_POS_RES.add(POS_RES);
		}

		if (mode != SttTagger.TagMode.POS)
			stats.print_res(descr + ".eval", CV, T);
		else {

			FileOutputStream fos = new FileOutputStream(descr + ".eval_pos");
			PrintStream out = new PrintStream(fos);
			for (int j = 0; j < T; ++j) {
				Vector<Double> all_j = new Vector<Double>();
				for (int i = 0; i < CV; ++i)
					all_j.add(CV_T_ALL.get(i).get(j));
				double mu_all_j = Stats.mean(all_j);
				double err_all_j = Stats.std_err(all_j, mu_all_j);
				out.println(mu_all_j + "\t" + err_all_j);
			}

			out.println();
			out.println();

			for (int r = 0; r < tagset.LSIS_size(); ++r)
				out.print(tagset.LSIS(r) + " ");
			out.println();

			for (int j1 = 0; j1 < T; ++j1) {
				for (int r = 0; r < tagset.LSIS_size(); ++r) {
					Vector<Double> pos_tj = new Vector<Double>();
					for (int i = 0; i < CV; ++i)
						pos_tj.add(CV_POS_RES.get(i).get(j1).get(r));
					double mu_tj = Stats.mean(pos_tj);
					out.print(mu_tj + " ");
				}
				out.println();
			}
			out.close();
		}
	}

	/**
	 * Train an SST model using the given train file and evaluate it on the
	 * given test file. Returns the evaluation results.
	 * 
	 * @param trainFileName
	 *            name of the file containing the training data.
	 * @param testFileName
	 *            name of the file containing the test data.
	 * @param encoding
	 *            encoding of the input files.
	 * @param numEpochs
	 *            number of epochs for the Perceptron trainning algorithm.
	 * @param wordWeight
	 *            word weight.
	 * 
	 * @return a vector with the results for each class (when there are more
	 *         than two classes) and the overall result.
	 * 
	 * @throws Exception
	 */
	public Vector<Verbose_res> trainAndTest(String trainFileName,
			String testFileName, String encoding, int numEpochs,
			Double wordWeight) throws Exception {
		Vector<Vector<Integer>> G = new Vector<Vector<Integer>>();
		Vector<Vector<Integer>> G2 = new Vector<Vector<Integer>>();
		Vector<String> ID = new Vector<String>();
		Vector<String> ID2 = new Vector<String>();

		Vector<Vector<Vector<Integer>>> D = new Vector<Vector<Vector<Integer>>>();
		Vector<Vector<Vector<Integer>>> D2 = new Vector<Vector<Vector<Integer>>>();
		load_data(trainFileName, encoding, D, G, ID, false, true);
		load_data(testFileName, encoding, D2, G2, ID2, false, false);

		String tagsetsuff, trainsuff, testsuff;

		trainsuff = new File(trainFileName).getName();
		testsuff = new File(testFileName).getName();
		tagsetsuff = tagset.getName();

		String description = tagsetsuff + "_" + trainsuff + "_" + testsuff
				+ ".results";

		Dataset cvTrainData = new SstDataset(D, G, ID);
		Dataset cvTestData = new SstDataset(D2, G2, ID2);

		return trainAndTest(cvTrainData, cvTestData, numEpochs, description,
				wordWeight);
	}

	/**
	 * Train an SST model and evaluate it on the given data.
	 * 
	 * @param cvTrainData
	 *            the trainning data.
	 * @param cvTestData
	 *            the testing data.
	 * @param numEpochs
	 *            number of epochs to train.
	 * @param descr
	 *            description of the model.
	 * @param wordWeight
	 *            word weight.
	 * 
	 * @return a vector with the results for each class and the overall result.
	 * 
	 * @throws Exception
	 */
	public Vector<Verbose_res> trainAndTest(Dataset cvTrainData,
			Dataset cvTestData, int numEpochs, String descr,
			Double wordWeight) throws Exception {

		// Create the algorithm object.
		PS_HMM ps_hmm_cv = newPSHMM(fb, tagset, ergodic);
		ps_hmm_cv.init(tagset.LSIS_size(), fb, false, wordWeight);

		// Train the algorithm along the given number of epochs (numEpochs).
		System.out.print("Iteration: 0");
		for (int t = 0; t < numEpochs; ++t) {
			ps_hmm_cv.train(cvTrainData);
			System.out.print("," + t);
		}
		System.out.println();

		// Apply the model to the given test data.
		int[][] tst_guess = ps_hmm_cv.guess_sequences(cvTestData);

		// Evaluate the predicted values.
		Vector<Verbose_res> tst_vr = new Vector<Verbose_res>();
		et.evaluate_sequences(cvTestData, tst_guess, tagset, LABELS.size(),
				tst_vr);

		return tst_vr;
	}

	/*
	 * R EVAL void eval_R(Zr D,Zr D2, Vector<Vector<Integer> > G,
	 * Vector<Vector<Integer> > G2, int T, int CV, String descr, String mode,
	 * Double ww, String thetafile){
	 * 
	 * Boolean no_special_symbol = false;
	 * 
	 * Myres res; res.init(CV,T,LABELS);
	 * 
	 * Vector<Vector<Double> > CV_T_ALL= new Vector<Vector<Double> >();
	 * Vector<Vector<Vector<Double> > > CV_POS_RES = new
	 * Vector<Vector<Vector<Double> > >();
	 * 
	 * Vector<Double> THETA = new Vector<Double>();
	 */
	/**
	 * LEER THETA ifstream in_theta(thetafile); char strbuf[100]; if
	 * (in_theta.is_open()){ THETA = new
	 * Vector<Double>(fb.LSIS.V_StringS.size()); while (in_theta.good()){ String
	 * y_str = ""; Double theta_y = 0; in_theta >> y_str >> theta_y; if (y_str
	 * != ""){ HM::iterator LSI_i = LSI.find(y_str); THETA[(*LSI_i).second] =
	 * theta_y; } in_theta.getline(strbuf,100); } }
	 * 
	 * System.err.print("\nTHETA = {"); for (int i = 0; i < THETA.size(); ++i)
	 * System.err.print(" "+ fb.LSIS.V_StringS.elementAt(i) + ":" + THETA[i];
	 * System.err.print(" }");
	 **/
	/**
	 * // NO BOOTSTRAP for (int cv = 0; cv < CV; ++cv){
	 * System.err.print("\ncv = "+ cv); Vector<Double> T_ALL = new
	 * Vector<Double>(); Vector<Vector<Double> > POS_RES= new
	 * Vector<Vector<Double> >(); PS_HMM ps_hmm_cv = new PS_HMM(); //? init
	 * ps_hmm_cv.init(int(LSI.size()),LIS,FIS,LSI,FSI,no_special_symbol, ww);
	 * 
	 * for (int t = 0; t < T; ++t){ ps_hmm_cv.train_R(D,G);
	 * Vector<Vector<Integer> > tst_guess = new Vector<Vector<Integer> >(); if
	 * (!THETA.isEmpty()) ps_hmm_cv.guess_sequences_R(D2,tst_guess,THETA); else
	 * ps_hmm_cv.guess_sequences_R(D2,tst_guess); double tst_f = 0; if (mode ==
	 * "POS"){ Vector<Double> itemized_res = new Vector<Double>(); tst_f =
	 * evaluate_pos(G2,tst_guess,LIS,LIS.size(),itemized_res);
	 * POS_RES.add(itemized_res); T_ALL.add(tst_f); } else { Vector<Verbose_res>
	 * tst_vr = new Vector<Verbose_res>(); tst_f =
	 * evaluate_sequences(G2,tst_guess,LIS,LIS.size(),tst_vr);
	 * res.update(cv,t,tst_f,tst_vr); } System.err.print("\ttst = "+tst_f); }
	 * CV_T_ALL.add(T_ALL); CV_POS_RES.add(POS_RES); }
	 * 
	 * if (mode != "POS") // NO BOOTSRAP res.print_res(descr+".eval",CV,T); else
	 * { FileOutputStream fos = new FileOutputStream(descr+".eval_pos");
	 * PrintStream out = new PrintStream(fos);
	 * 
	 * 
	 * for (int j = 0; j < T; ++j){ Vector<Double> all_j = new Vector<Double>();
	 * for (int i = 0; i < CV; ++i)
	 * all_j.add(CV_T_ALL.elementAt(i).elementAt(j)); Double mu_all_j =
	 * Stats.mean(all_j); Double err_all_j = Stats.std_err(all_j,mu_all_j);
	 * out.println(mu_all_j + "\t" + err_all_j); }
	 * 
	 * out.println(); out.println(); for (int r = 0; r <
	 * fb.LSIS.V_StringS.size(); ++r) out.print(fb.LSIS.get(r)+" ");
	 * out.println();
	 * 
	 * for (int j = 0; j < T; ++j){ for (int r = 0; r <
	 * fb.LSIS.V_StringS.size(); ++r){ Vector<Double> pos_tj = new
	 * Vector<Double>(); for (int i = 0; i < CV; ++i)
	 * pos_tj.add(CV_POS_RES.elementAt(i).elementAt(j).elementAt(r)); double
	 * mu_tj = Stats.mean(pos_tj); out.print(mu_tj+ " "); } out.println(); }
	 * out.close();
	 * 
	 * } }
	 **/

	/**
	 * A method for training a model
	 * 
	 * @train TrainingData
	 * @T number of epochs
	 * @modename Name of the output model
	 * @throws FileNotFoundException
	 **/
	public void train(Dataset train, int T, String modelname)
			throws FileNotFoundException {

		Double ww = 1.0;
		boolean no_special_symbol = false;
		ps_hmm.init(tagset.LSIS_size(), fb, no_special_symbol, ww);
		for (int t = 1; t <= T; ++t)
			ps_hmm.train(train); // ps_hmm.train(D,G);
		ps_hmm.SaveModel(modelname, fb);
	}

	/**
	 * 
	 * A method to train the parser
	 * 
	 * @param modelname
	 * @param traindata
	 * @param tagsetname
	 * @param secondorder
	 * @param T
	 * @param mode
	 * @throws Exception
	 */
	public void train_light(String modelname, FeatureBuilder fb,
			String traindata, String encoding, SstTagSet tagset,
			boolean secondorder, int T) throws Exception {
		logger.info("\ntrain-light(" + "\n\tmodelname = " + modelname
				+ "\n\ttraindata = " + traindata + "\n\ttagsetname = "
				+ tagset.getName() + "\n\tsecondorder = " + secondorder
				+ "\n\tT = " + T + "\n\tmode = " + mode + " )");
		// SstTagSet stagset = new SstTagSet(tagsetname,"UTF-8");
		SttTagger TL = new SttTagger(fb, tagset, ergodic);
		// TL.init("NULL",tagsetname,mode,fb);
		Dataset TD = TL.load_data(traindata, encoding, secondorder);
		TL.train(TD, T, modelname);// TL.train(TD.D,TD.G,T,modelname);
	}

}
