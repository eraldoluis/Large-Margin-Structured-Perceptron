package tagger.features;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.HashMap;

import org.apache.log4j.Logger;

import tagger.utils.FileDescription;
import tagger.utils.Utils;


/**
 * @author jordi
 * 
 * an object for doing morphological analysis using a cache
 */
public class MorphCache {
	
private static final String MORPH_ENCODING = "UTF-8";

static Logger logger = Logger.getLogger(MorphCache.class);
	
/// Two hashmap POS => HASH<word form => lemma>	
HashMap<String, HashMap<String,String> > morph_cache;

/// Maximum PoS Length
int posMaxLength=10;

///get WordNet lemma
public  String get_lemma(String wl,String pos){
	 //@TODO cache is lower case 
	 if (pos.length() > posMaxLength) //JAB FIX
	      pos = pos.substring(0,posMaxLength);
	  HashMap<String,String> morph=morph_cache.get(pos);
	  if (morph!=null){
	    String w_i = morph.get(wl);
	    if (w_i != null) {
	    	return w_i;
	    }
	    else { 
	    	wl =  wl.toLowerCase(); 
	    	w_i = morph.get(wl);
		    if (w_i != null) {
		    	return w_i;
		    }	
	    }
	  }
	  else{
	    logger.warn("PoS >"+pos+"< not found");
	  }
	  
	  return wl;
 }


public MorphCache(FileDescription file) throws FileNotFoundException, IOException {
	morph_cache= new HashMap<String, HashMap<String,String> >();
	load(file.path,file.encoding);
}

void load(String filename,String morphEncoding) throws FileNotFoundException, IOException {
	load(Utils.getBufferedReader(filename,morphEncoding));
}

void load(BufferedReader fin) throws IOException {
	String input;
while ((input=fin.readLine())!=null){
    
    String [] buff= input.split("[\t ]");
    String pos = buff[0];
    String w = buff[1];
    String wm = buff[2];
    
    if (pos.length() > posMaxLength) //JAB FIX
      pos = pos.substring(0,posMaxLength);
    
    //This is quite inconsistent with the way we look up things
      String wl = w.toLowerCase();
    if (wl != wm){
    	HashMap<String,String>val = morph_cache.get(pos);
      if (val==null){
	    HashMap<String,String> tmp=new HashMap<String,String>();
		tmp.put(wl,wm);
		morph_cache.put(pos,tmp);
      }
      else {
    	  String wl_val = val.get(wl);
    	  if (wl_val==null)
    		  val.put(wl,wm);
      	}
    }
  }
  System.err.println("\t|M| = "+morph_cache.size());
}

int size() { return morph_cache.size();}

}