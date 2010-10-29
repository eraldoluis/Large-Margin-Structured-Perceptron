package tagger.extra;


import org.junit.After;
import org.junit.Before;
import org.junit.Test;



import junit.framework.TestCase;

public class TagsetParoleTest extends TestCase{
	TagsetParole tagset;
	@Before
	public void setUp() throws Exception {
		tagset= new TagsetParole();
		tagset.loadMap("/home/jordi/parole.txt");
	}

	@After
	public void tearDown() throws Exception {
	}
	
	@Test
	public void testTag2Features() {
		
		assertEquals("test aq perdurable", tagset.tag2Features("aq0cp0"), "num=p|gen=c"); 
		assertEquals("test aq deteriorat", tagset.tag2Features("aq0fpp"), "num=p|gen=f|fun=p"); 
		assertEquals("test da la", tagset.tag2Features("da0fs0"),"num=s|gen=f");
		assertEquals("test vm estan",  tagset.tag2Features("vmip3p0"),     "num=p|per=3|mod=i|ten=p");
		assertEquals("test nc suícidi", tagset.tag2Features("ncms000"), "num=s|gen=m");
		assertEquals("test sp de",	tagset.tag2Features("sps00"), "for=s");
		assertEquals("test sp dels", tagset.tag2Features("spcmp"),"num=p|gen=m|for=c");
        assertEquals("test p0 ell",tagset.tag2Features("p0300000"), "per=3");
        assertEquals("test pp ella", tagset.tag2Features("pp3fs000"),"num=s|per=3|gen=f");
	
        
        assertEquals("test vm és", "num=s|per=3|mod=i|ten=p",tagset.tag2Features("vmip3s0"));
        assertEquals("test vm comptarà", "num=s|per=3|mod=i|ten=f",tagset.tag2Features("vmif3s0"));
        assertEquals("test va  va", "num=s|per=3|mod=i|ten=p", tagset.tag2Features("vaip3s0"));
        assertEquals("test va  van", "num=p|per=3|mod=i|ten=p", tagset.tag2Features("vaip3p0"));
        assertEquals("test pd  això",  "num=s|gen=n",tagset.tag2Features("pd0ns000"));
	
        assertEquals("test pi quelcom","num=n|gen=c" ,tagset.tag2Features("pi0cn000"));
         
        assertEquals("test di un", "num=s|gen=m",tagset.tag2Features("di0ms0"));
        assertEquals("test pi un", "num=s|gen=m",tagset.tag2Features("pi0ms000"));
        assertEquals("test pn un", "num=s|gen=m",tagset.tag2Features("pn0ms000"));
        assertEquals("test pp (pol) vostès",  "num=s|pol=p|per=2|gen=c", tagset.tag2Features("pp2cs00p"));
        assertEquals("test pp vostè",  "num=p|pol=p|per=2|gen=c", tagset.tag2Features("pp2cp00p"));
        
        assertEquals("test pp (cas) li",   "num=s|cas=d|per=3|gen=c", tagset.tag2Features("pp3csd00"));
        assertEquals("test pr que","num=n|gen=c", tagset.tag2Features("pr0cn000"));
        assertEquals("test pt què",   "num=s|gen=c", tagset.tag2Features("pt0cs000"));
        
        assertEquals("test px seu",    "num=s|per=3|gen=m", tagset.tag2Features("px3ms000"));
        
        //WEAR parole does not provide gender?? num=s|per=3|pos=s and pos is disordered
        assertEquals("test px (pos) la_meva",     "num=s|per=3|pos=s|gen=f", tagset.tag2Features("px3fs0s0"));
	
        assertEquals("test dp (pos) la_nostra","num=s|per=1|pos=p|gen=f",tagset.tag2Features("dp1fsp"));
        
        assertEquals("test dn cinc","num=p|gen=c", tagset.tag2Features("dn0cp0"));
        // Wear it only appears once
        assertEquals("test de Que",     "num=n|gen=c", tagset.tag2Features("de0cn0"));
        assertEquals("test dd aquesta", "num=s|gen=f", tagset.tag2Features("dd0fs0"));
        
	}
}
