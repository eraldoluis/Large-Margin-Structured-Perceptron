package tagger.features;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.Vector;
import java.util.HashMap;

import tagger.data.FeatureValueEncoding;
import tagger.data.ModelDescription;
import tagger.data.SstTagSet;
import tagger.utils.FileDescription;
import tagger.utils.Utils;

/**
 * 
 * A class to provide feature base on a gazetter (list of names)
 * 
 * the constructor needs a file containing the names of the files to load
 * 
 * @TODO provide the files encoding
 * 
 * Features generated from Gazetter
 * 
 * TRIG
 * RX
 * LX
 * B-
 * I-
 * 
 * Uses a morph cache to lemmatize
 * 
 * @author jordi
 *
 */
public class Gazetter extends FeatureBuilderBasic {
	private static final String FEATURE_PREFIX_LX = "LX-";
	private static final String FEATURE_PREFIX_RX = "RX-";
	private static final String FEATURE_PREFIX_TRIGGER = "TRIG";
	private static final String GAZ_ENCODING = "UTF-8";
	
	
	private static org.apache.log4j.Logger log = org.apache.log4j.Logger.getLogger(Gazetter.class.getName());

	/// LEMMA => Encoded Features
	HashMap<String,Vector<Integer> > gazetter;
	
	boolean ready;
	
	/// Maximum span
	int maxspan;

	/// Morphological Cache (needed for lemmatizing)
	MorphCache M;

	/**
	 * 
	 * @param basename
	 * @param fname
	 * @param encoding
	 * @param compress
	 * @param p_maxspan
	 * @param model
	 * @param tagset
	 * @param morphCache
	 * @param morphEncoding
	 * @throws Exception
	 */
	public Gazetter(String basename,
			        String fname,
			        int p_maxspan,
			        ModelDescription model, 
			        FileDescription morphFile) throws Exception { 
		super(model);
		M = new MorphCache(morphFile);
		init(basename,fname,p_maxspan);
	}
	/**
	 * 
	 * @param basename
	 * @param fname
	 * @param encoding
	 * @param compress
	 * @param p_maxspan
	 * @param model
	 * @param tagset
	 * @param mcache
	 * @throws Exception
	 */
	public Gazetter(String basename, String fname,ModelDescription model,int p_maxspan,MorphCache mcache) throws Exception {
		super(model);;
		M =mcache;
		init(basename,fname,p_maxspan);
	}
	
	void Ginit(int p_maxspan) {
		ready=false; maxspan=p_maxspan; 
		gazetter = new HashMap<String,Vector<Integer> >(); 
	}

	





	/**
	 * Load a gazetter from the file named fname
	 * 
	 * @param fname
	 * @return
	 * @throws IOException
	 */
	Vector<String> load_GAZ_files(String fname, String encoding) throws IOException{
		Vector<String> GAZ = new Vector<String>();
		BufferedReader gazFilenames = Utils.getBufferedReader(fname , encoding);
		String gazFile;
		while((gazFile=gazFilenames.readLine())!=null) {
			GAZ.add(gazFile);
		}

		log.info("\t|G("+fname+")| = "+GAZ.size());
		return GAZ;
	}


	void init(String basename, String fname, int n) throws IOException { 
		Ginit(n);
		if (!ready){
			log.info("\n\tgaz_server::init("+fname+","+n+")");
			maxspan = n;
			Vector<String> fnames = load_GAZ_files(fname,GAZ_ENCODING);
			for (int i = 0; i < fnames.size(); ++i){
				BufferedReader fgaz = Utils.getBufferedReader(basename+fnames.get(i),GAZ_ENCODING);
				String line;
				while ( (line=fgaz.readLine())!=null){
					String [] buff= line.split("[\t]");
					String w = buff[0].toLowerCase();

					// register gazzetter features
					Vector<Integer> tags = new Vector<Integer>();
					for(int j=1;j<buff.length;++j){
						tags.add(FSIS.add_update_hmap(buff[j]));
					}

					// store the list of features associated to the entry
					if (tags.size()>0)
						update_G(w,tags);
				}
			}
			
			log.info("\t|G| = "+gazetter.size());
			ready = true;
		}
	}



	void update_G(String w,Vector<Integer> tags) {
		Vector<Integer> elem = gazetter.get(w);
		if(elem==null) 
			gazetter.put(w,tags);
		else 
			for (int i = 0; i < tags.size(); ++i)
				gazetter.get(w).add(tags.get(i));
	}




	/**
	 * @BUG
	 * We should return lemma's list 
	 * lemma's feature should be provided (if requested) but they are  calculated after call basic extractFeatures (??) 
	 *
	 * extract features should be redesigned?
	 */
	public  Vector<String> getLemmasVector(String[] L, String[] P) {
		Vector<String> WM = new Vector<String>();
		int _L_ = L.length;
		for (int i = 0; i < _L_; ++i){
			WM.add(M.get_lemma(L[i], P[i]));
		}
		return WM;
	}

	public  String[] getLemmas(String[] L, String[] P) {
		int _L_ = L.length;
		String[] WM = new String[_L_];

		for (int i = 0; i < _L_; ++i){
			WM[i]=(M.get_lemma(L[i], P[i]));
		}
		return WM;
	}

	public    Vector<Vector<String> >  extractFeatures(String[] W,String[] P,String[] L) {
		//@TODO fix jab WM lemma list is not returned?
		if(L.length<W.length) {L = getLemmas(W,P);}

		Vector<Vector<String> >  newFeatures = super.extractFeatures(W,P,L);


		InternalExtract_feats(FSIS, W, P, newFeatures, L);
		return  newFeatures;
	}

	void InternalExtract_feats(FeatureValueEncoding LIS, String[] L, String[] P,Vector<Vector<String> > O,String[] WM) {
		int _L_ = L.length;

		//??JAB if (O.size()>0)  O.resize(_L_);

		for (int i = 0; i < _L_; ++i){

			//@autor JAB add a previous call to calculate lemmas
			//if (M.size()>0) //??JAB EMPTY??
			//	  WM.add(get_wm(L[i], P[i].substring(0,Math.min(2,P[i].length()))));

			for (int j = 0; j < maxspan; ++j)
				if (i-j >= 0){
					int lx = i-j;
					String w = L[lx], wm = "", pos = P[lx];

					if (M.size()>0)//??JAB EMPTY??
						wm = WM[lx].toLowerCase();

					++lx;
					while (lx <= i){
						w += "_"+L[lx];
						wm += "_"+WM[lx].toLowerCase();;
						++lx;
					}

					if (w.length()-j+1 > 2){


						Vector<Integer> G_i= gazetter.get(w);

						if (G_i==null) G_i= gazetter.get(wm);

						if (G_i!=null){
							for (int r = 0; r < G_i.size(); ++r){
								int left = i-j;
								int G_i_r_id=G_i.get(r);
								boolean isTrigger = LIS.get(G_i_r_id).startsWith(FEATURE_PREFIX_TRIGGER);
								String G_i_r_Label = LIS.get(G_i_r_id);
								if (isTrigger)
									if (left-1 >= 0)
										O.get(left-1).add(FEATURE_PREFIX_RX+G_i_r_Label);
								O.get(left++).add("B-"+G_i_r_Label);
								while (left <= i)
									O.get(left++).add("I-"+G_i_r_Label);
								if (isTrigger && left < _L_)
									O.get(left).add(FEATURE_PREFIX_LX+G_i_r_Label);
							}
						}
					}
				}
		}

	}  


}
