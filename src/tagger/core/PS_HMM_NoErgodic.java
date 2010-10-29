package tagger.core;

import java.io.IOException;

import tagger.data.ModelDescription;
import tagger.data.SstTagSet;
import tagger.features.FeatureBuilder;

/**
 * Non ergodic PS_HMM
 * 
 * @author jordi
 * 
 */
public class PS_HMM_NoErgodic extends PS_HMM {
	// / States are +1 shift as we neet to represent initial transition
	public boolean allow[][];
	protected int INIT = 0;
	FeatureBuilder fb;

	public PS_HMM_NoErgodic(FeatureBuilder fb, SstTagSet tagset) {
		super(tagset);
		this.fb = fb;
	}

	// @Override
	public void jabLoadModel(ModelDescription modelname) throws IOException {
		super.jabLoadModel(modelname);
		init(k);
	}

	// @Override
	public void init(int k_val, FeatureBuilder fb, boolean no_spec, double wc) {
		super.init(k_val, fb, no_spec, wc);
		this.fb = fb;
		init(k_val);
	}

	void init(int k_val) {
		allow = new boolean[k_val + 1][k_val];
		INIT = k_val;
		int idLabelI = 0;
		for (String labelI : tagset.getListLabel()) {

			int idLabelJ = 0;
			for (String labelJ : tagset.getListLabel()) {

				if (labelI.charAt(0) == 'I') {
					allow[idLabelI][idLabelJ] = (labelJ.charAt(0) == '0'
							|| idLabelI == idLabelJ || labelJ.charAt(0) == 'B');
				} else {
					if (labelI.charAt(0) == '0') {
						allow[idLabelI][idLabelJ] = (labelJ.charAt(0) == '0' || labelJ
								.charAt(0) == 'B');
					} else { // starts with B
						allow[idLabelI][idLabelJ] = (labelJ.charAt(0) == '0'
								|| labelJ.charAt(0) == 'B' || labelI
								.compareTo("B-" + labelJ.substring(2)) == 0);
					}
				}
				idLabelJ++;
			}
			allow[INIT][idLabelI] = (labelI.charAt(0) == '0' || labelI
					.charAt(0) == 'B');
			idLabelI++;
		}
		// Trace the table :)
		int i = 0;
		for (String labelI : tagset.getListLabel()) {
			System.err.println("allow[-1," + labelI + "]=" + allow[INIT][i]);
			int j = 0;
			for (String labelJ : tagset.getListLabel()) {
				System.err.println("allow[" + labelI + "," + labelJ + "]="
						+ allow[i][j]);
				++j;
			}

			++i;
		}
	}

	// @Override
	protected void viterbi_tj(double[][] delta, int[] x, int t, int j,
			int[][] psi) {
		try {
			double wc = getEmissionParameter(x, j, t);
			// TRACE PrintStream out = new PrintStream(new
			// FileOutputStream("kk.err",true));
			// TRACE out.println("WC["+j+","+t+"]="+wc);
			if (t == 0) {
				psi[t][j] = -1;
				if (allow[INIT][j])
					delta[t][j] = wc + getFinalStateParameter(j);
				else
					delta[t][j] = -Double.MAX_VALUE;
				return;
			}

			double max = Double.NEGATIVE_INFINITY; // -1e100;
			int argmax = 0;
			for (int i = 0; i < k; ++i) {
				double score_ij = -Double.MAX_VALUE;
				if (allow[i][j])
					score_ij = delta[t - 1][i] + wc + getTransitionParameter(i, j);

				// TRACE out.println("delta["+(t-1)+","+i+"]="+delta[t-1][i]);
				// TRACE out.println("get_phi_b["+i+","+j+"]="+get_phi_b( i, j
				// ));
				if (score_ij > max) {
					max = score_ij;
					argmax = i;
				}
				// TRACE out.println("max "+max+" argmax "+argmax);
			}
			psi[t][j] = argmax;
			delta[t][j] = max;
			// TRACE out.close();
		} catch (Exception e) {
			e.printStackTrace();
		}
	}

}
