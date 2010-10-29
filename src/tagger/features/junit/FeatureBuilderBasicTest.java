package tagger.features.junit;

import static org.junit.Assert.*;

import java.util.Locale;

import junit.framework.TestCase;

import org.junit.Test;

import tagger.features.FeatureBuilderBasic;

public class FeatureBuilderBasicTest extends TestCase {
	
	private FeatureBuilderBasic fb; 
	
	
	public FeatureBuilderBasicTest() {
		super();
	}
	
	protected void setUp() { 
	      try {
			//fb = new FeatureBuilderBasic("bechmarks/conll/XX.TAGSET");
		} catch (Exception e) {
			
			e.printStackTrace();
		}
	   }
	
	@Test
	public void testMy_tolower_sh() {
		// Get default locale
	    Locale locale = Locale.getDefault();
	    
	    // Set the default locale to pre-defined locale
	    Locale.setDefault(Locale.UK);
	    
	    // Set the default locale to custom locale
	    locale = new Locale("en", "UK");
	    Locale.setDefault(locale);
		StringBuilder sh = new StringBuilder();
		FeatureBuilderBasic.my_tolower_sh("AXxAB",sh); 
		
		assertEquals("ascii  AxxAB",sh.toString(),"XxX");
		sh = new StringBuilder();
	//	FeatureBuilderBasic.my_tolower_sh("AÁaá",sh);
	//	assertEquals("nascii AÁaá",sh.toString(),"Xx");
		
	}

}
