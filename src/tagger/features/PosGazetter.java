package tagger.features;

import java.util.Vector;

import tagger.core.SttTagger;
import tagger.data.ModelDescription;
import tagger.data.SstTagSet;
import tagger.data.SstTagSetPOS;
import tagger.utils.FileDescription;

/**
 * A feature Builder that extends a Gazetter and adds PoS
 * 
 * @author jordi
 * 
 * @TODO POS model should be configurable??
 */
public class PosGazetter extends Gazetter {
	 SttTagger PoSTagger;
	 FeatureBuilderBasic fb;
	 
	public PosGazetter(String basename, String fname, ModelDescription model,ModelDescription posModel,int p_maxspan,
			FileDescription morphFile) throws Exception {
		super(basename, fname, p_maxspan, model, morphFile);
		// fb could be the same or union with the BIO MODEL?
	
		//@TODO hardwired MODEL
		//  fb = new  FeatureBuilderBasic(basename+"MODELS/WSJPOS_up_17", encoding, compress);
		//  SstTagSet tagset = new SstTagSetPOS(basename+"DATA/WSJPOS.TAGSET","UTF-8");
		//  PoSTagger= new  SttTagger(fb,new ModelDescription(basename+"MODELS/WSJPOS_up_17", tagset, "UTF-8", compress), true);
		
		  fb = new  FeatureBuilderBasic(posModel);
		  PoSTagger= new  SttTagger(fb,posModel, true);
		
	}
	
/// @TODO remove parameter EMPTYpos
	  public    Vector<Vector<String> >  extractFeatures(String[] W,String[] EMPTYpos) {
		   String[] fvpos = {};
		   String[] vpos = PoSTagger.tagSequence(W, fvpos,new String[0]);
		   Vector<Vector<String> >  newFeatures= super.extractFeatures(W,vpos, new String[0]); //empty lemma list

	      return  newFeatures;
	  }
}
