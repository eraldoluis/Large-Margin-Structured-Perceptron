package tagger.features.junit;

import java.util.Vector;

import org.junit.Before;
import org.junit.Test;

import tagger.core.SttTagger;
import tagger.data.ModelDescription;
import tagger.data.SstTagSet;
import tagger.data.SstTagSetBIO;
import tagger.features.FeatureBuilderBasic;
import tagger.features.Gazetter;

import junit.framework.TestCase;

public class SttTaggerTest extends TestCase {
     SttTagger tagger; 
     
	@Before
	protected void setUp() throws Exception {
		
		
		SstTagSet tagset = new SstTagSetBIO("/home/y/share/nlr/sst/TAGSETS/CONLL03.TAGSET","UTF-8");
		FeatureBuilderBasic bfb = new FeatureBuilderBasic(
				new ModelDescription("/home/y/share/nlr/sst/MODELS/WSJCONLL_base_gaz10_up_25", tagset,"UTF-8", false));
		
		 tagger = new SttTagger(bfb,tagset,true); 
	}
	
	@Test
	public void testProcessLine() {
	  String line=	"1\tpos+1=CD w+1=1996-08-30 sh+1=d-d-d sb+1=30 w=london pos=NNP sh=X pr=lo pr3=lon sb=on sb3=don B-WORLD_1 B-LOC\tpos-1=NNP w-1=london sh-1=X sb-1=on w=1996-08-30 pos=CD sh=d-d-d pr=19 pr3=199 sb=30 sb3=-30 0";
	  Vector<Vector<Vector<Integer> > > D = new Vector<Vector<Vector<Integer> > >();
	     Vector<Vector<Integer> > G = new Vector<Vector<Integer> > ();
	     Vector<String> ID = new Vector<String>();
	     tagger.processLine(line, D, G, ID, false,false);
	     System.err.println("Line Processed");
	     
	     Vector<String> EID = new Vector<String>(); EID.add("1");
	     Vector<Vector<Integer> > EG = new Vector<Vector<Integer> > ();
	     Vector<Integer> v= new  Vector<Integer>();
	     v.add(tagger.tagset.LSIS_add_update_hmap("B-LOC"));
	     v.add(tagger.tagset.LSIS_add_update_hmap("0"));
	     EG.add(v);
	     Vector<Vector<Vector<Integer> > > ED = new Vector<Vector<Vector<Integer> > >();
	     Vector<Vector<Integer> > vD = new Vector<Vector<Integer> > ();
	     Vector<Integer> vf = new Vector<Integer>();
	     String feat[] = {"pos+1=CD", "w+1=1996-08-30","sh+1=d-d-d","sb+1=30",
	     "w=london",
	     "pos=NNP",
	     "sh=X",
	     "pr=lo",
	     "pr3=lon",
	     "sb=on",
	     "sb3=don",
	     "B-WORLD_1"};
	     
	     for(String sv: feat) {
	    	 Integer iv = tagger.fb.FSIS_update_hmap(sv);
	    	 if(iv!=null) vf.add(iv);
	    	 else {System.err.println("feature "+sv+" skip");}
	         
	     }
	     vD.add(vf);
	     vf = new Vector<Integer>();
	     String featd[] ={
	     "pos-1=NNP",
	     "w-1=london",
	     "sh-1=X",
	     "sb-1=on",
	     "w=1996-08-30",
	     "pos=CD",
	     "sh=d-d-d",
	     "pr=19",
	     "pr3=199",
	     "sb=30",
	     "sb3=-30"};
	     for(String sv: featd) {
	    	 Integer iv = tagger.fb.FSIS_update_hmap(sv);
	    	 if(iv!=null) vf.add(iv);
	    	 else {System.err.println("feature "+sv+" skip");}
	         
	     }  
	     
	     
	     vD.add(vf);
	     ED.add(vD);
	     System.err.println("EG"+EG);
	     System.err.println("G"+G);
	     System.err.println("D"+D);
	     System.err.println("ED"+ED);
	     assertEquals("testid",ID, EID); 
	     assertEquals("testG",G, EG);
	     assertEquals("TestD",D,ED);
	}
}
