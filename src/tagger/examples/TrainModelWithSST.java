package tagger.examples;

import tagger.core.SttTagger;
import tagger.data.SstTagSet;
import tagger.data.SstTagSetBIO;
import tagger.features.FeatureBuilderBasic;

/**
 * An example of training a new model.
 * 
 */
public class TrainModelWithSST {
	public static void main(String[] args) throws Exception {

		// If you want to replicate the experiment you can set the random seed
		// e.g.
		// learnTagger.ps_hmm.randGen=new Random(1144556677);

		// T (number of epoch, i.e. iterations) must be provided (could be found
		// using eval see FindTparamWithSST)
		// int T=10;
		// learnTagger.train_light("necatmodel",
		// "benchmarks/ancora/ancora_alltrain.feat",
		// "benchmarks/conll/CONLL03.TAGSET", false, T, "BIO");
		// learnTagger.train_light("nelemacatmodel",
		// "/home/jordi/Escriptori/carlos/trainingPLNiso.feat",
		// "benchmarks/conll/CONLL03.TAGSET", false, T, "BIO");

		int T = 14;
		FeatureBuilderBasic fb = new FeatureBuilderBasic();
		SstTagSet tagset = new SstTagSetBIO("benchmarks/conll/CONLL03.TAGSET",
				"UTF-8");
		SttTagger learnTagger = new SttTagger(fb, tagset, true);
		learnTagger.train_light("allnelemacatmodel", fb,
				"/home/jordi/Escriptori/carlos/alltrainingPLNiso.feat",
				"UTF-8", tagset, false, T);
	}
}
