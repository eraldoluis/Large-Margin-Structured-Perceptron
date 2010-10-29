package tagger.examples;

import java.util.Random;

import tagger.core.SttTagger;
import tagger.data.SstTagSet;
import tagger.data.SstTagSetBIO;
import tagger.features.FeatureBuilderBasic;
import tagger.features.Gazetter;

public class FindTparamWitSST {

	public static void main() throws Exception {
	  
	 // If you want to replicate the experimetn you can set the random seed e.g.
	 //learnTagger.ps_hmm.randGen=new Random(1144556677);
	 //learnTagger.eval_light("benchmarks/conll/CONLL03.BI_trn.feat", "benchmarks/conll/CONLL03.BI_dev.feat", "benchmarks/conll/CONLL03.TAGSET", false, 10, 4, "BIO", false, 1.0, "");
	 FeatureBuilderBasic fb = new  FeatureBuilderBasic();
	 SstTagSet tagset = new SstTagSetBIO("benchmarks/conll/CONLL03.TAGSET", "UTF-8");
	 SttTagger learnTagger= new  SttTagger(fb,tagset,true);
	 learnTagger.eval_light("/home/jordi/Escriptori/carlos/trainingPLNiso.feat", "/home/jordi/Escriptori/carlos/devPLNiso.fea", "UTF-8", fb, tagset , false, 10, 4, false, 1.0, "");
		
	} 
}
