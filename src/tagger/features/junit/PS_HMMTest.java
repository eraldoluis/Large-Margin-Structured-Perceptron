/**
 * 
 */
package tagger.features.junit;

import java.util.Arrays;

import junit.framework.TestCase;
import tagger.core.PS_HMM;
import tagger.data.SstTagSetBIO;

/**
 * @author jordi atserias
 * 
 */
public class PS_HMMTest extends TestCase {

	private PS_HMM pshmm;

	/**
	 * @param name
	 */
	public PS_HMMTest(String name) {
		super(name);
	}

	protected void setUp() {
		// @TODO fix tagset issue!
		try {
			pshmm = new PS_HMM(new SstTagSetBIO(
					"/home/y/share/nlr/sst/TAGSETS/CONLL03.TAGSET", "UTF-8"));
		} catch (Exception e) {
			e.printStackTrace();
		}
	}

	/**
	 * Test method for {@link tagger.core.HmmTrainer#rand_reorder_vector(int[])}
	 * .
	 */
	public final void testRand_reorder_vector() {
		int[] test1 = { 1, 2, 3, 4, 5 };
		int[] out = PS_HMM.rand_reorder_vector(test1);
		Arrays.sort(out);
		assertTrue("array resort equality", Arrays.equals(test1, out));
	}

}
