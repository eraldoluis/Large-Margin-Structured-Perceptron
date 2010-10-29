package tagger.examples;

import tagger.core.SttTagger;
import tagger.data.SstTagSet;
import tagger.data.SstTagSetBIO;
import tagger.data.Dataset;
import tagger.features.FeatureBuilderBasic;

public class SplitTrainingDev {
	public static void main(String args[]) throws Exception {
	 FeatureBuilderBasic fb = new  FeatureBuilderBasic();
	 SstTagSet tagset = new SstTagSetBIO("benchmarks/conll/CONLL03.TAGSET","UTF-8");
	 SttTagger learnTagger= new  SttTagger(fb,tagset,true);
	 Dataset trainingdata;
	 String traindataFile="/home/jordi/Escriptori/carlos/trainingPLNiso.feat";
	 boolean secondOrder=false;
	 trainingdata = learnTagger.load_data(traindataFile,"UTF-8",secondOrder);
	 Dataset[] traincv= trainingdata.splitTrain(10);
	 trainingdata =null;
	 
	 learnTagger.eval_light(traincv[0], traincv[1], fb, tagset, false, 15, 4, false, 1.0, "");
	}
}
