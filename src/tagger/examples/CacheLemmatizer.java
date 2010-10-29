package tagger.examples;

import tagger.features.MorphCache;
import tagger.utils.FileDescription;

/**
 * An example about using the morphological cache for lemmatizing
**/
public class CacheLemmatizer {
	
	  
	public static void main(String args[]) throws Exception {
		MorphCache M= new MorphCache(new FileDescription("/home/y/share/nlr/sst/DATA/MORPH_CACHE","UTF-8",false));
		System.err.println("cats lemma:"+M.get_lemma("cats", "NNS"));
	}
}
