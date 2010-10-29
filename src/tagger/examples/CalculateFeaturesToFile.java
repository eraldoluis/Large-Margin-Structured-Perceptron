package tagger.examples;

import tagger.data.ModelDescription;
import tagger.data.SstTagSet;
import tagger.data.SstTagSetBIO;
import tagger.features.FeatureBuilderBasic;
import tagger.features.Gazetter;
import tagger.features.MorphCache;
import tagger.utils.FileDescription;

public class CalculateFeaturesToFile {
	public static void main(String args[]) throws Exception {
	 // FeatureBuilderBasic fb = new  FeatureBuilderBasic("benchmarks/conll/CONLL03.TAGSET");
	 // fb.tagfile("benchmarks/ancora/ne.catdev.txt","benchmarks/ancora/ancora_dev.feat","ISO-8859-1", "UTF-8",false,false);
	 // fb.tagfile("benchmarks/ancora/ne.cattrain.txt","benchmarks/ancora/ancora_trn.feat","ISO-8859-1", "UTF-8",false,false);
	 // fb.tagfile("benchmarks/ancora/ne.cattst.txt","benchmarks/ancora/ancora_tst.feat","ISO-8859-1", "UTF-8",false,false);
	 // fb.tagfile("/home/jordi/Escriptori/carlos/trainingPLNiso.txt","/home/jordi/Escriptori/carlos/trainingPLNiso.feat","ISO-8859-1", "UTF-8",true,true);
	 // fb.tagfile("/home/jordi/Escriptori/carlos/testPLNiso.txt","/home/jordi/Escriptori/carlos/testPLNiso.feat","UTF-8", "UTF-8",true,true);
		String morphCache="/home/y/share/nlr/sst/DATA/MORPH_CACHE";
		MorphCache M = new MorphCache(new FileDescription(morphCache,"UTF-8",false));
		
		SstTagSet tagset = new SstTagSetBIO("/home/y/share/nlr/sst/TAGSETS/WNSS_07.TAGSET","UTF-8");
		Gazetter fb = new  Gazetter("","/home/y/share/nlr/sst/DATA/gazlistall_minussemcor", //"UTF-8", false,
				new ModelDescription("/home/y/share/nlr/sst/MODELS/SEM07_base_gaz10_up_12", tagset, "UTF-8", false),
				4,
				M);
	 	fb.USE_LEMMA=true;
	fb.tagfile("/home/jordi/benchmarks/CONLL03.BI_dev","/home/jordi/benchmarks/CONLL03.BI_dev.feat","UTF-8", "UTF-8",true,false);
	}
}


