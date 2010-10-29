package tagger.learning;

/***
 *  STILL SOMEM ERRORS PASING INTEGERS AS RETURN PARAMETERS!!!
 *  LOOK FOR @JAB
 */
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.PrintStream;
import java.util.HashMap;
import java.util.Set;
import java.util.Vector;

import tagger.utils.Pair;

public class LearningStatistics {

	// / matrix _CV_ x _T_
	public double M[][];

	// / mean on M
	public Vector<Double> MU = new Vector<Double>();

	// / std_err on M
	public Vector<Double> STD = new Vector<Double>();

	// / Label =>
	public HashMap<String, Inclass_statistics> M_verbose = new HashMap<String, Inclass_statistics>();

	public void dumpM_verbose(PrintStream cout) {
		for (String mkey : M_verbose.keySet()) {
			cout.print("M_verbose[" + mkey + "]= (");
			M_verbose.get(mkey).dump(cout);
			cout.println(") =M_verbose[" + mkey + "]");
		}
	}

	public static void dumpVectorDouble(PrintStream cout, String name,
			Vector<Double> v) {
		for (int i = 0; i < v.size(); ++i)
			cout.println(name + "[" + i + "]=" + v.get(i).doubleValue());
	}

	public static void dumpArraydouble(PrintStream cout, String name,
			double[][] a) {
		for (int i = 0; i < a.length; ++i)
			for (int j = 0; j < a.length; ++j)
				cout.println(name + "[" + i + "," + j + "]=" + a[i][j]);
	}

	public void dump(PrintStream cout) {
		dumpM_verbose(cout);
		dumpArraydouble(cout, "M", M);
		dumpVectorDouble(cout, "MU", MU);
		dumpVectorDouble(cout, "STD", STD);
	}

	public void init(int CV, int T) {
		M = new double[CV][T];
	}

	public void init(int CV, int T, Set<String> LABELS) {
		init(CV, T);

		java.util.Iterator<String> iter = LABELS.iterator();
		while (iter.hasNext()) {
			M_verbose.put(iter.next(), new Inclass_statistics(CV, T));
		}
	}

	public void update(int cv_i, int t_j, double upd_val) {
		M[cv_i][t_j] = upd_val;
	}

	/**
	 * update statistics: M and M_verbose
	 * 
	 * @param cv_i
	 * @param t_j
	 * @param upd_val
	 * @param vr
	 * @throws Exception
	 */
	public void update(int cv_i, int t_j, double upd_val, Vector<Verbose_res> vr)
			throws Exception {
		System.err.println("\nmyres::update |vr| = " + vr.size());

		PrintStream cout = new PrintStream(new FileOutputStream("cout.txt",
				true));
		// TRACE
		for (int i = 0; i < vr.size(); ++i) {
			cout.print("vr[" + i + "]=");
			vr.get(i).dump(cout);
		}
		// TRACE
		cout.println("cv_i=" + cv_i + " t_j=" + t_j + " upd_val=" + upd_val);

		M[cv_i][t_j] = upd_val;

		int _vr_ = vr.size();
		for (int i = 0; i < _vr_; ++i) {
			Inclass_statistics Mv_i = M_verbose.get(vr.get(i).L);

			if (Mv_i == null) {
				throw new Exception("\nM_verbose.find(" + vr.get(i).L
						+ ") = NULL");
			} else {
				// @JAB ?? looks like a copy
				Mv_i.M_nobj[cv_i][t_j] = (double) vr.get(i).nobjects;
				Mv_i.M_nans[cv_i][t_j] = (double) vr.get(i).nanswers;
				Mv_i.M_nfull[cv_i][t_j] = (double) vr.get(i).nfullycorrect;
				Mv_i.M_R[cv_i][t_j] = vr.get(i).getRecall();
				Mv_i.M_P[cv_i][t_j] = vr.get(i).getPrecision();
				Mv_i.M_F[cv_i][t_j] = vr.get(i).getF1();
			}
		}

		this.dump(cout);
		cout.close();
	}

	Pair<Double, Double> stats_M(double[][] ds, int t) {
		Vector<Double> scores_s = new Vector<Double>();
		int _cv_ = ds.length;
		for (int i = 0; i < _cv_; ++i)
			scores_s.add(ds[i][t]);

		Double mu = Stats.mean(scores_s);
		Double std = Stats.std_err(scores_s, mu);
		return new Pair<Double, Double>(mu, std);
	}

	public void stats(int CV, int T) {
		MU.clear();
		STD.clear();
		for (int i = 0; i < T; ++i) {
			Vector<Double> RES_i = new Vector<Double>();
			for (int j = 0; j < CV; ++j) {
				RES_i.add(M[j][i]);
			}
			Double mu_i = Stats.mean(RES_i);
			Double std_i = Stats.std_err(RES_i, mu_i);
			MU.add(mu_i);
			STD.add(std_i);
		}
	}

	public void print_res(String fname) throws FileNotFoundException {
		PrintStream out = new PrintStream(new FileOutputStream(fname));
		int _M_ = M.length, _m_ = M[0].length;
		stats(_M_, _m_);
		Vector<Vector<Integer>> MAXS = new Vector<Vector<Integer>>();
		for (int i = 0; i < _M_; ++i) {
			Vector<Double> M_i = new Vector<Double>();
			for (int j = 0; j < _m_; ++j) {
				out.print(M[i][j] + " ");
				M_i.add(M[i][j]);
			}
			out.println();
			Vector<Integer> MAX_i = new Vector<Integer>();
			Xmax(M_i, MAX_i);
			MAXS.add(MAX_i);
		}
		out.println();
		for (int i = 0; i < _M_; ++i) {
			int _max_ = MAXS.get(i).size();
			for (int j = 0; j < _max_; ++j)
				out.print(MAXS.get(i).get(j) + " ");
			out.println();
		}
		out.println();
		for (int j = 0; j < _m_; ++j)
			out.println(MU.get(j) + " " + STD.get(j));
	}

	/**
	 * prints a description of the results
	 * 
	 * @param fname
	 * @param CV
	 * @param T
	 * @throws FileNotFoundException
	 */
	public void print_res(String fname, int CV, int T)
			throws FileNotFoundException {

		System.err.println("\nprint_res(" + fname + ")");

		FileOutputStream fos = new FileOutputStream(fname);
		PrintStream out = new PrintStream(fos);

		int _M_ = M.length, _m_ = M[0].length;
		stats(_M_, _m_);
		Vector<Vector<Integer>> MAXS = new Vector<Vector<Integer>>();
		for (int i = 0; i < _M_; ++i) {
			Vector<Double> M_i = new Vector<Double>();
			for (int j = 0; j < _m_; ++j) {
				out.print(M[i][j] + " ");
				M_i.add(M[i][j]);
			}
			out.println();
			Vector<Integer> MAX_i = new Vector<Integer>();
			Xmax(M_i, MAX_i);
			MAXS.add(MAX_i);
		}
		out.println();
		for (int i = 0; i < _M_; ++i) {
			int _max_ = MAXS.get(i).size();
			for (int j = 0; j < _max_; ++j)
				out.print(MAXS.get(i).get(j) + " ");
			out.println();
		}
		out.println();
		for (int j = 0; j < _m_; ++j)
			out.println(MU.get(j) + " " + STD.get(j));

		// / VERBOSE ///
		out.println("\nVERBOSE:");
		for (int i = 0; i < T; ++i) {
			out.println("\nT = " + i + "\n----------------\n");
			for (java.util.Iterator<String> Mv_i = M_verbose.keySet()
					.iterator(); Mv_i.hasNext();) {
				String ekey = Mv_i.next();
				Inclass_statistics econtent = M_verbose.get(ekey);
				// @TODO JAB ??? initilizations ?? VALUES WILL NOT BE
				// RETURNED!!!
				// Double mu_nobj = null, std_nobj = null, mu_nans= null,
				// std_nans= null, mu_nfull= null, std_nfull= null, mu_R= null,
				// std_R= null, mu_P= null, std_P= null, mu_F= null, std_F=
				// null;
				Pair<Double, Double> nobj = stats_M(econtent.M_nobj, i);
				Pair<Double, Double> nans = stats_M(econtent.M_nans, i);
				Pair<Double, Double> nfull = stats_M(econtent.M_nfull, i);
				Pair<Double, Double> R = stats_M(econtent.M_R, i);
				Pair<Double, Double> P = stats_M(econtent.M_P, i);
				Pair<Double, Double> F = stats_M(econtent.M_F, i);
				out.println("\n" + ekey + "\tnobj = " + nobj.first + "/"
						+ nobj.second + "\tnans = " + nans.first + "/"
						+ nans.second + "\tnfull = " + nfull.first + "/"
						+ nfull.second + "\tR = " + R.first + "/" + R.second
						+ "\tP = " + P.first + "/" + P.second + "\tF = "
						+ F.first + "/" + F.second);
			}
		}
	}

	void Xmax(Vector<Double> X, Vector<Integer> MAX) {
		int max_i = 0, _X_ = X.size();
		double max_val = 0;
		for (int i = 0; i < _X_; ++i)
			if (X.get(i) >= max_val) {
				max_i = i;
				max_val = X.get(i);
			}
		for (int i = 0; i < _X_; ++i)
			if (X.get(i) == max_val)
				MAX.add(i);
	}

}
