package tagger.core;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintStream;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.Random;
import java.util.Set;
import java.util.Vector;

import tagger.data.ModelDescription;
import tagger.data.SstTagSet;
import tagger.data.Dataset;
import tagger.data.SstDatasetExample;
import tagger.features.FeatureBuilder;
import tagger.features.Fs;
import tagger.utils.Utils;

public class PS_HMM extends HmmModel {

	private static final String MODELS_PATH = "MODELS/";
	public static final String FEATURE_MODEL_FILENAME_EXTENSION = ".F";
	public static final String PS_HMM_MODEL_FILENAME_EXTENSION = ".PS_HMM";
	private static final String MODEL_ENCODING = "UTF-8";

	// /To allow debugging-reproduceable results using the same random seed
	public static Random randGen = new Random();

	// / @JAB only for internal debugging to limit the trace
	static int nit = 0;

	// / size of state space/total number of iterations
	int it;

	// / Average mode on updating weights
	boolean av_mod;

	// / when true, only use active features
	boolean only_active;

	// / the ratio of word_contribution/label-label contr
	double word_weight;

	// / temporary variables useful for carrying around
	int cur_ex;

	// /
	SstTagSet tagset;

	protected int cur_pos;

	/**
	 * Initial state parameters.
	 * 
	 * The value <code>phi_s[i]</code> is the parameter associated with the
	 * probability of starting at the state <code>i</code>. One may not use this
	 * parameters. Instead, one may use the state 'zero' as the initial state.
	 */
	protected Fs[] phi_s;

	/**
	 * Final state parameters.
	 * 
	 * The value <code>phi_f[i]</code> is the parameter associated with the
	 * probability of ending at the state <code>i</code>. One may not use this
	 * parameters. Instead, one may use the state 'zero' as the final state.
	 */
	protected Fs[] phi_f;

	/**
	 * Transition parameters.
	 * 
	 * The value <code>phi_b[i][j]</code> is the parameter associated with the
	 * transition from state/label <code>i</code> to state/label <code>j</code>.
	 */
	protected Fs[][] phi_b;

	/**
	 * Emission weights.
	 * 
	 * The value <code>phi_w[i].get(f)</code> is the weight to emit the feature
	 * <code>f</code> in state <code>i</code>.
	 */
	Vector<HashMap<Integer, Fs>> phi_w;

	// /jordi: active features (to filter and use only certain features, this
	// option is controled by only_active)
	Vector<HashSet<Integer>> active_phi_w;

	public PS_HMM(SstTagSet tagset) {
		super();
		this.tagset = tagset;
		it = 0;
		only_active = false;
		av_mod = true;
		cur_ex = -1;
		word_weight = 1;

		phi_w = new Vector<HashMap<Integer, Fs>>();
		active_phi_w = new Vector<HashSet<Integer>>();

		// @TODO control phi initilization (load model vs constructor, k needs
		// to be known)
		// phi_s = new Fs[k];
		// phi_f= new Fs[k];
		// phi_b = new Fs[k][k];
	}

	public void init(int k_val, FeatureBuilder fb, boolean no_spec, double wc) {
		logger.info("\nPS_HMM::init(" + k_val + ")");
		k = k_val;
		word_weight = wc;
		no_special_end_transition_sym = no_spec;
		if (!no_spec) {
			// ?Constructor
			phi_s = new Fs[k];
			// ?Constructor
			phi_f = new Fs[k];
			init_phi_sf(k, phi_s, fb, true, tagset); // bigrams begin
			init_phi_sf(k, phi_f, fb, false, tagset); // bigrams end
			logger.info("Phi lengh" + phi_s.length);
		}
		// ?Constructor
		phi_b = new Fs[k][k];
		init_phi_b(k, phi_b, fb, tagset); // std transitions
		init_phi_w(k, phi_w, fb.FSIS_size(), active_phi_w);
		// sparse matrix for std features
		logger.info("\tdone");
	}

	/***
	 * This 4 functions is what chain needs
	 * 
	 */

	@Override
	protected double getFinalStateParameter(int j) {
		return phi_s[j].avg_val(it, av_mod);
	}

	@Override
	protected double getInitialStateParameter(int j) {
		return phi_f[j].avg_val(it, av_mod);
	}

	@Override
	protected double getTransitionParameter(int i, int j) {
		return phi_b[i][j].avg_val(it, av_mod);
	}

	@Override
	protected double getEmissionParameter(int[] x, int j, int pos) {
		double word_contribution = 0;
		int _x_ = x.length;
		if (only_active)
			for (int i = 0; i < _x_; ++i) {
				if (active_phi_w.get(j).contains(x[i])) {
					Fs mFs = phi_w.get(j).get(x[i]);
					if (mFs != null)
						word_contribution += mFs.avg_val(it, av_mod);
				}
			}
		else
			for (int i = 0; i < _x_; ++i) {
				Fs mFs = phi_w.get(j).get(x[i]);
				if (mFs != null)
					word_contribution += mFs.avg_val(it, av_mod);
			}
		return word_weight * word_contribution;
	}

	// vector duality
	double get_word_contribution(Vector<Integer> x, int j, int pos) {
		double word_contribution = 0;
		int _x_ = (int) x.size();
		if (only_active)
			for (int i = 0; i < _x_; ++i) {
				if (active_phi_w.get(j).contains(x.get(i))) {
					Fs mFs = phi_w.get(j).get(x.get(i));
					if (mFs != null)
						word_contribution += mFs.avg_val(it, av_mod);
				}
			}
		else
			for (int i = 0; i < _x_; ++i) {
				Fs mFs = phi_w.get(j).get(x.get(i));
				if (mFs != null)
					word_contribution += mFs.avg_val(it, av_mod);
			}
		return word_weight * word_contribution;
	}

	/*
	 * VECTOR IMPLEMENTATIONS
	 * 
	 * void viterbi_tj( Vector<Vector<Double> > delta, Vector<Integer> vector,
	 * int t, int j, Vector<Vector<Integer> > psi ) { double wc =
	 * get_word_contribution( vector, j, t ); if ( t == 0 ){
	 * psi.get(t).insertElementAt(-1,j); delta.get(t).insertElementAt(wc +
	 * get_phi_s( j ),j) ; return; } double max = Double.MIN_VALUE; // -1e100;
	 * int argmax = 0; for (int i = 0; i < k; ++i){ double score_ij =
	 * delta.get(t-1).get(i) + wc + get_phi_b( i, j ); if (score_ij > max ){ max
	 * = score_ij; argmax = i; } } psi.get(t).insertElementAt(argmax,j) ;
	 * delta.get(t).insertElementAt(max,j); }
	 * 
	 * Vector<Integer> viterbi(Vector<Vector<Integer>> vector) { Vector<Integer>
	 * U = new Vector<Integer>(); int N = (int) vector.size(); int argmax_t = 0;
	 * 
	 * Vector<Vector<Double> > delta = new Vector<Vector<Double> >((int) N);//(
	 * N, vector<double>( k, 0.0 ) ); for(int i=0;i<N;++i) delta.add(new
	 * Vector<Double>(k)); Vector<Vector<Integer> > psi = new
	 * Vector<Vector<Integer> >((int) N);//( N, vector<Integer>( k, 0 ) );
	 * for(int i=0;i<N;++i) psi.add(new Vector<Integer>(k)); // ATTENTION!!! //
	 * Incorporate these lines into semisupervised code // if( mode == Train )
	 * // avr = false; for( int t = 0; t < N; ++t ) for ( int j = 0; j < k; ++j
	 * ) viterbi_tj( delta, vector.get(t), t, j, psi ); double max_t =
	 * Double.MIN_VALUE;//-1e100; for (int j = 0; j < k; ++j){ double last_bit =
	 * delta.get(N-1).get(j) + get_phi_f( j ); if ( last_bit > max_t ){ max_t =
	 * last_bit; argmax_t = j; } } decode_s( U, delta, psi, argmax_t ); return
	 * U; }
	 */

	void decode_s(Vector<Integer> _u, Vector<Vector<Double>> delta,
			Vector<Vector<Integer>> psi, int argmax_t) {
		int N = delta.size();
		int best = argmax_t;
		/*
		 * _u.insert( _u.begin(), argmax_t );
		 * 
		 * for ( int t = N - 1 ; t > 0 ; --t ){ _u.insert( _u.begin(),
		 * psi.get(t).get(best) ); best = psi.get(t).get(best); }
		 */
		Vector<Integer> r_u = new Vector<Integer>();
		r_u.add(argmax_t);
		for (int t = N - 1; t > 0; --t) {
			best = psi.get(t).get(best);
			r_u.add(best);
		}

		// @JAB
		for (int t = r_u.size() - 1; t >= 0; --t) {
			_u.add(r_u.get(t));
		}

	}

	/**
	 * Load Model
	 * 
	 * @param modelname
	 * @throws IOException
	 */
	public void jabLoadModel(ModelDescription model) throws IOException {
		String modelname = model.path;
		logger.info("\nLoadModel(" + modelname + ")");
		modelname += PS_HMM_MODEL_FILENAME_EXTENSION;
		if (model.compress)
			modelname += Utils.GZIP_FILENAME_EXTENSION;
		jabLoadModel(Utils.getBufferedReader(modelname, model.encoding));
	}

	void jabLoadModel(BufferedReader fin) throws IOException {
		// @JAB jabLoad_XIS(modelname+".F");
		logger.info(".");

		// char strbuf[1000];
		int _phis_, _phif_, _phib_, _phiw_;

		// in >> k >> it >> _phis_ >> word_weight;
		k = Integer.parseInt(fin.readLine());
		phi_s = new Fs[k];
		phi_f = new Fs[k];
		phi_b = new Fs[k][k];

		it = Integer.parseInt(fin.readLine());
		_phis_ = Integer.parseInt(fin.readLine());
		word_weight = Double.parseDouble(fin.readLine().split("[\t]")[0]);

		// in.getline(strbuf,1000);
		phi_s = new Fs[_phis_];
		for (int i = 0; i < _phis_; ++i) {
			// Fs fs;
			// in >> fs.alpha >> fs.avg_num >> fs.lst_upd >> fs.upd_val;
			Fs fs = Fs.parseFs(fin.readLine());
			phi_s[i] = fs;
		}
		// in >> _phif_;
		_phif_ = Integer.parseInt(fin.readLine().split("[ \t]")[0]);
		// in.getline(strbuf,1000);
		logger.info(".");
		for (int i = 0; i < _phif_; ++i) {
			// Fs fs;
			// in >> fs.alpha >> fs.avg_num >> fs.lst_upd >> fs.upd_val;
			Fs fs = Fs.parseFs(fin.readLine());
			phi_f[i] = fs;
		}
		// in >> _phib_;
		_phib_ = Integer.parseInt(fin.readLine().split("[ \t]")[0]);
		// in.getline(strbuf,1000);
		logger.info(".");
		String buff[];
		int bp;
		phi_b = new Fs[_phib_][];
		for (int i = 0; i < _phif_; ++i) {
			int _i_;
			// in >> _i_;

			buff = fin.readLine().split("[\t ]");
			bp = 0;

			_i_ = Integer.parseInt(buff[bp++]);
			Fs[] Fi = new Fs[_i_];
			for (int j = 0; j < _i_; ++j) {
				// Fs fs;
				// in >> fs.alpha >> fs.avg_num >> fs.lst_upd >> fs.upd_val;
				Fs fs = new Fs(Double.parseDouble(buff[bp++]),
						Double.parseDouble(buff[bp++]),
						Integer.parseInt(buff[bp++]),
						Double.parseDouble(buff[bp++]));

				Fi[j] = fs;
			}
			phi_b[i] = Fi;
		}
		// in >> _phiw_;
		_phiw_ = Integer.parseInt(fin.readLine().split("[ \t]")[0]);
		// in.getline(strbuf,1000);
		logger.info(".");

		for (int i = 0; i < _phiw_; ++i) {

			buff = fin.readLine().split("[\t ]");
			bp = 0;

			int _i_ = Integer.parseInt(buff[bp++]);
			HashMap<Integer, Fs> Fi = new HashMap<Integer, Fs>();
			// while (in.peek() == 9){
			while (bp < buff.length) {
				Fs fs;
				int fid;
				// in >> fid >> fs.alpha >> fs.avg_num >> fs.lst_upd >>
				// fs.upd_val;
				fid = Integer.parseInt(buff[bp++]);
				fs = new Fs(Double.parseDouble(buff[bp++]),
						Double.parseDouble(buff[bp++]),
						Integer.parseInt(buff[bp++]),
						Double.parseDouble(buff[bp++]));
				Fi.put(fid, fs);
			}
			phi_w.add(Fi);
		}
		String check;
		check = fin.readLine();
		if (!check.startsWith("ENDOFMODEL")) {
			logger.error("\n\tcheck error:" + check);
			throw new IOException("Format error on laoding model ");
		} else
			logger.info("\tOK");
	}

	//
	// methods call from evaluation training
	//
	/*
	 * VECTOR IMPLEMENTATION void
	 * guess_sequences(Vector<Vector<Vector<Integer>>> d2,
	 * Vector<Vector<Integer> > EVAL_L ){ int _E_ = d2.size(); int tenth =
	 * (int)(_E_/ (double)(10)); if (tenth < 1) tenth = 1; for ( int i = 0; i <
	 * _E_; ++i ){ Vector<Integer> GUESS_i = viterbi( d2.get(i)); EVAL_L.add(
	 * GUESS_i ); if (i%tenth==0)System.err.print("."); } }
	 */

	//
	// methods call from evaluation training
	// R versions
	/*
	 * public void guess_sequences_R( Zr EVAL_D, Vector<Vector<Integer> > EVAL_L
	 * ){ int _E_ = EVAL_D.size(); int tenth = (int)(_E_/(float)(10)); if (tenth
	 * < 1) tenth = 1; for ( int i = 0; i < _E_; ++i ){ Vector<Integer> GUESS_i;
	 * viterbi_R( EVAL_D.elementAt(i), GUESS_i ); EVAL_L.add( GUESS_i ); if
	 * (i%tenth==0) System.err.print("."); }
	 * 
	 * 
	 * ////// PRIORS on guess
	 * 
	 * public void guess_sequences_R( Zr EVAL_D, Vector<Vector<Integer> >
	 * EVAL_L, Vector<Double> THETA ){ int _E_ = EVAL_D.size(); int tenth =
	 * (int)(_E_/(float)(10)); if (tenth < 1) tenth = 1; for ( int i = 0; i <
	 * _E_; ++i ){ Vector<Integer> GUESS_i; viterbi_R( EVAL_D.elementAt(i),
	 * GUESS_i, THETA ); EVAL_L.add( GUESS_i ); if (i%tenth==0)
	 * System.err.print("."); } }
	 */

	/**
	 * Train one epoch on a vector of examples.
	 * 
	 * Train just one epoch with the given examples. Randomly reorder the
	 * examples before training. This method can be called several times, one
	 * for each epoch.
	 * 
	 * 
	 * @param trainset
	 */
	public void train(Dataset trainset) {
		int N = trainset.size();
		Set<Integer> Index = new HashSet<Integer>();
		for (int i = 0; i < N; ++i)
			Index.add(i);
		train(trainset, Index);
		// train( W, T, Index );
	}

	/**
	 * jordi: random reorder of a vector (probalby can be better reimplemented)
	 * 
	 * @param IN
	 * @return
	 */
	public static int[] rand_reorder_vector(int[] IN) {
		int N = IN.length;
		int[] OUT = new int[N];
		logger.info("\trand_reorder(|IN|=" + N + ",|OUT|=");
		Set<Integer> visited = new HashSet<Integer>();
		int j = 0;
		while (j < N) {
			// jordi: use random instead of Math.random() so we can use a
			// reproduce random seq (same seed)
			int s = (int) (Math.floor(randGen.nextDouble() * N));
			if (!visited.contains(s)) {
				visited.add(s);
				OUT[j++] = IN[s];
			}
		}
		logger.info(OUT.length + ")");
		return OUT;
	}

	/**
	 * Train one epoch on a vector of examples.
	 * 
	 * Train just one epoch with the given examples. Randomly reorder the
	 * examples before training. The vector <code>Index</code> contains the
	 * indexes of the examples to be used in the training.
	 * 
	 * @param tokens
	 *            example vector.
	 * @param tags
	 *            the associated correct labels.
	 * @param Index
	 *            vector with the indexes of the examples to be used in this
	 *            training. This vector is also used to randomly reorder the
	 *            examples.
	 */
	// void train( int[][][] W, int[][] T, Set<Integer> Index ){
	void train(Dataset trainset, Set<Integer> Index) {
		int updates = 0, counter = 0, N = Index.size();
		logger.info("\ntrain(|Index| = " + N + ")");
		int[] IN = new int[N];
		int[] OUT;
		int j = 0;
		for (Iterator<Integer> I_i = Index.iterator(); I_i.hasNext();)
			IN[j++] = (Integer) I_i.next();

		// TO TRACE with NORANDOM for ( int i=0;i<Index.size();++i) { IN[i]=i;}

		OUT = rand_reorder_vector(IN);

		for (int i = 0; i < N; ++i) {
			++it;
			cur_ex = OUT[i];
			// @TODO jab training corpus should provide a way to get a single
			// example
			// boolean updated_s = train_on_s( W[OUT[i]], T[OUT[i]]);
			boolean updated_s = train_on_s(trainset.getExample(OUT[i]));

			// TRACE NORANDOM cur_ex = IN[i];
			// TRACE NORANDOM boolean updated_s = train_on_s( W[IN[i]],
			// T[IN[i]]);
			if (updated_s)
				++updates;
			if (++counter % 1000 == 0)
				System.err.print(".");
		}
		logger.info("\ti = " + it + "\ts-error = " + (double) updates / N);
	}

	/**
	 * Train on one example.
	 * 
	 * @param e
	 *            the example.
	 * 
	 * @return true if PS_HMM has to be updated (? a.k.o. lazzy evaluation of
	 *         the PS_HMM weight).
	 */
	boolean train_on_s(SstDatasetExample e) {
		int[][] W = e.tokens;
		int[] T = e.tags;
		try {
			av_mod = false;

			// TRACE String filename = String.format("train%04d.jlog",nit);
			// nit++;
			// TRACE System.err.println("Log:"+filename);
			// TRACE FileWriter tlog = new FileWriter(filename);

			int[] _u = viterbi(W);

			// @TODO check-find out why av_mod needs to be changed true/false
			// (?)
			av_mod = true;

			// trace imput
			// TRACE for(int i=0;i<T.length;++i)
			// TRACE tlog.write(i+":"+T[i]+" "+_u[i]+"\n");

			boolean s_update = update(W, T, _u);

			// trace return and state
			/*
			 * TRACE int i=0; for(Fs e: phi_s) {
			 * tlog.write("phi_s["+i+"]="+e.dump()+"\n"); ++i; }
			 * 
			 * i=0; for(Fs e: phi_f) {
			 * tlog.write("phi_f["+i+"]="+e.dump()+"\n"); ++i; }
			 * 
			 * i=0; for(Fs[] le: phi_b) { int j=0; for(Fs e: le){
			 * tlog.write("phi_b["+i+","+j+"]="+e.dump()+"\n"); ++j; } ++i; }
			 * 
			 * 
			 * // size is k but only some elements used (?) int p=0;
			 * for(HashMap<Integer,Fs> e: phi_w) { Integer sk[] =
			 * e.keySet().toArray(new Integer[1]);
			 * 
			 * Arrays.sort(sk); for(Integer kk: sk) { if(kk != null)
			 * tlog.write("phi_w["+p+"]="+e.get(kk)+"\n"); } ++p; }
			 * 
			 * 
			 * // we are non using active_phi but initilizing it
			 * for(HashSet<Integer> ap: active_phi_w) { Integer sk[] =
			 * ap.toArray(new Integer[1]); Arrays.sort(sk); for(Integer kk: sk)
			 * { if(kk != null) tlog.write("active_phi_w  = "+k+"\n"); }
			 * 
			 * }
			 * 
			 * 
			 * 
			 * tlog.write("update return "+ s_update+"\n"); tlog.close();
			 */

			cur_ex = -1; // ??
			return s_update;
		} catch (Exception e1) {
			e1.printStackTrace();
			return false;
		}
	}

	/**
	 * Update the model parameters for a given example.
	 * 
	 * @param W
	 *            the example.
	 * @param T
	 *            the correct label sequence for this example.
	 * @param U
	 *            the label sequence predicted by the current model.
	 * @return
	 */
	boolean update(int[][] W, int[] T, int[] U) {
		// Sentence length.
		int sentLen = U.length;

		boolean s_update = false;

		// Update first and last states.
		update_boundary_feats(U[0], T[0], U[sentLen - 1], T[sentLen - 1]);

		// Update the emission parameters of the first-word features.
		if (U[0] != T[0]) {
			s_update = true;
			cur_pos = 0;
			update_word_feats(W[0], T[0], U[0]);
		}

		// Update the transition and word parameters.
		for (int i = 1; i < sentLen; ++i)
			if (U[i] != T[i]) {
				s_update = true;
				cur_pos = i;
				update_word_feats(W[i], T[i], U[i]);
				update_transition_feats(T[i - 1], T[i], U[i - 1], U[i]);
			}

		// Finalize the updates. This is due to average perceptron.
		if (s_update) {
			if (U[0] != T[0])
				finalize_update_word_feats(W[0], T[0], U[0], s_update);
			for (int i = 1; i < sentLen; ++i)
				if (U[i] != T[i])
					finalize_update_word_feats(W[i], T[i], U[i], s_update);
			finalize_transition_feats(s_update);
		}

		return s_update;
	}

	protected void finalize_update_word_feats(int[] W, int T, int U, boolean su) {
		int _X_ = W.length;
		for (int s = 0; s < _X_; ++s) {
			finalize_feature_val(phi_w, U, W[s], it, only_active, active_phi_w,
					su);
			finalize_feature_val(phi_w, T, W[s], it, only_active, active_phi_w,
					su);
		}
	}

	void finalize_feature_val(Vector<HashMap<Integer, Fs>> phi_w, int l,
			int feat, int it, boolean active,
			Vector<HashSet<Integer>> act_phi_w, boolean s_upd) {
		// hashSet<Integer>::iterator active_F_i = act_phi_w[l].find( feat );
		if (!active || act_phi_w.get(l).contains(feat)) {
			Fs v = phi_w.get(l).get(feat);
			assert (v != null);
			// System.err.println("before:"+phi_w.get(l).get( feat ).dump());
			v.update(it, s_upd);
			// System.err.println("update:"+phi_w.get(l).get( feat ).dump());
		}
	}

	/**
	 * Update the emission values associated with all the features of the
	 * current token.
	 * 
	 * @param w
	 *            the features present in the current token/word.
	 * @param T
	 *            the correct state/label for this token/word.
	 * @param U
	 *            the state/label predicted by the current model.
	 */
	protected void update_word_feats(int[] w, int T, int U) {
		int _X_ = w.length;
		for (int s = 0; s < _X_; ++s) {
			update_feature_val(phi_w, U, w[s], -1, only_active, active_phi_w);
			update_feature_val(phi_w, T, w[s], 1, only_active, active_phi_w);
		}
	}

	/**
	 * Update the value of the emission weight
	 * <code>phi_w.get(label).get(feat)</code>.
	 * 
	 * Update the weight associated with the state/label <code>label</code> and
	 * the word/feature <code>feat</code>, i.e.,
	 * <code>phi_w.get(label).get(feat)</code>. If there is no <code>feat</code>
	 * in the <code>label</code> label dictionary, then insert a value for this
	 * label/state in the dictionary.
	 * 
	 * @param phi_w
	 *            the matrix of emission weights.
	 * @param label
	 *            the state (or label).
	 * @param feat
	 *            the feature (or word).
	 * @param val
	 *            the value to be added to the corresponding weight.
	 * @param active
	 *            if <code>true</code> then use the feature filter
	 *            <code>active_phi_w</code> that contains the active features.
	 * @param active_phi_w
	 *            the matrix of active features among all the available
	 *            features.
	 */
	void update_feature_val(Vector<HashMap<Integer, Fs>> phi_w, int label,
			int feat, double val, boolean active,
			Vector<HashSet<Integer>> active_phi_w) {

		if (active && !active_phi_w.get(label).contains(feat)) // was not active
																// till now, add
																// as
																// active
			active_phi_w.get(label).add(feat);

		Fs f;
		if (!phi_w.get(label).containsKey(feat)) {
			f = new Fs();
			phi_w.get(label).put(feat, f);
		} else
			f = phi_w.get(label).get(feat);

		f.upd_val += val;
	}

	/**
	 * Update the transition weights <code>phi_b[bT][eT]</code> and
	 * <code>phi_b[bU][eU]</code>.
	 * 
	 * The weight of the transition <code>phi_b[bT][eT]</code> is incremented
	 * (correct transition that is missing in the current example) and the
	 * weight of the transition <code>phi_b[bU][eU]</code> is decremented (wrong
	 * transition that is present in the current example).
	 * 
	 * @param bT
	 * @param eT
	 * @param bU
	 * @param eU
	 */
	protected void update_transition_feats(int bT, int eT, int bU, int eU) {
		phi_b[bT][eT].upd_val += 1;
		phi_b[bU][eU].upd_val -= 1;
	}

	/**
	 * 
	 * TODO: ERROR CHECK THAT UPDATE is not returned!!!
	 * 
	 * @param s_update
	 */
	protected void finalize_transition_feats(boolean s_update) {
		int bs = phi_b.length;
		for (int i = 0; i < bs; ++i)
			for (int j = 0; j < bs; ++j)
				phi_b[i][j].update(it, s_update);
		if (no_special_end_transition_sym)
			for (int i = 0; i < bs; ++i) {
				phi_s[i].equate(phi_b[0][i]);
				phi_f[i].equate(phi_b[i][0]);
			}
		else
			for (int i = 0; i < bs; ++i) {
				phi_s[i].update(it, s_update);
				phi_f[i].update(it, s_update);
			}
	}

	protected void update_boundary_feats(int bU, int bT, int eU, int eT) {
		if (no_special_end_transition_sym) {
			phi_b[0][bU].upd_val -= 1;
			phi_b[0][bT].upd_val += 1;
			phi_b[eU][0].upd_val -= 1;
			phi_b[eT][0].upd_val += 1;
		} else {
			phi_s[bU].upd_val -= 1;
			phi_s[bT].upd_val += 1;
			phi_f[eU].upd_val -= 1;
			phi_f[eT].upd_val += 1;
		}
	}

	/*
	 * @TODO those could be set as static methods?
	 * 
	 * @eraldo: I think they could, but shouldn't :).
	 */
	static void init_phi_b(int k, Fs[][] phi_b, FeatureBuilder fb,
			SstTagSet ptagset) {
		logger.info("\n\tinit_phi_b(" + k + "): ");

		for (int tlx = 0; tlx < k; ++tlx) {
			Fs[] _Fb = new Fs[k];
			for (int trx = 0; trx < k; ++trx) {
				Fs f = new Fs();
				String id_str = ptagset.LSIS(tlx) + "-" + ptagset.LSIS(trx);
				// @TODO check what's up with f_id is not uset, should we
				// register it
				int f_id = fb.FSIS_update_hmap(id_str, true); // f_id = 0;
				_Fb[trx] = f;
			}
			phi_b[tlx] = _Fb;
		}
		logger.info(" |phi_b| = " + phi_b.length + "*" + phi_b[0].length);
	}

	static void init_phi_sf(int k, Fs[] phi_sf, FeatureBuilder fb,
			boolean is_s, SstTagSet ptagset) {
		// cerr << "\n\tinit_phi_sf(" << is_s << ")";

		for (int r = 0; r < k; ++r) {
			String id_str = ptagset.LSIS(r);
			if (is_s)
				id_str = "#-" + id_str;
			else
				id_str = id_str + "-#";
			Fs f = new Fs();
			// @TODO guess what is up ?? f_id is not used
			// ?? should we register it
			int f_id = fb.FSIS_update_hmap(id_str, true); // f_id = 0;
			phi_sf[r] = f;
		}
		// cerr << " |phi_sf| = " << phi_sf.size();
	}

	static void init_phi_w(int k, Vector<HashMap<Integer, Fs>> phi_w, int _wf_,
			Vector<HashSet<Integer>> active_phi_w) {
		logger.info("\n\tphi_w(" + k + ")");

		for (int i = 0; i < k; ++i) {
			HashMap<Integer, Fs> tmp = new HashMap<Integer, Fs>();
			phi_w.add(tmp);
			HashSet<Integer> stmp = new HashSet<Integer>();
			active_phi_w.add(stmp);
			logger.info("\t|phi_w| = " + phi_w.size());
		}
	}

	/**
	 * Write the model
	 * 
	 * @param modelname
	 * @param fb
	 * @throws FileNotFoundException
	 */
	public void SaveModel(String modelname, FeatureBuilder fb)
			throws FileNotFoundException {
		logger.info("\nSaveModel(" + modelname + ")");
		PrintStream out = new PrintStream(MODELS_PATH + modelname
				+ PS_HMM_MODEL_FILENAME_EXTENSION);
		print_XIS_XSI(MODELS_PATH + modelname
				+ FEATURE_MODEL_FILENAME_EXTENSION, fb);

		out.println(k + "\n" + it + "\n" + phi_s.length + "\n" + word_weight
				+ "\tphi_s\talpha\tavg_num\tlst_upd\tupd_val");
		for (int i = 0; i < phi_s.length; ++i)
			out.println(phi_s[i].alpha + " " + phi_s[i].avg_num + " "
					+ phi_s[i].lst_upd + " " + phi_s[i].upd_val);
		out.println(phi_f.length + "\tphi_f");
		for (int i = 0; i < phi_f.length; ++i)
			out.println(phi_f[i].alpha + " " + phi_f[i].avg_num + " "
					+ phi_f[i].lst_upd + " " + phi_f[i].upd_val);
		out.println(phi_b.length + "\t" + "\tphi_b");
		for (int i = 0; i < phi_b.length; ++i) {
			int _i_ = phi_b[i].length;
			out.print(_i_);
			for (int j = 0; j < _i_; ++j)
				out.print("\t" + phi_b[i][j].alpha + " " + phi_b[i][j].avg_num
						+ " " + phi_b[i][j].lst_upd + " " + phi_b[i][j].upd_val);
			out.println();
		}
		out.println(phi_w.size() + "\tphi_w");
		for (int i = 0; i < phi_w.size(); ++i) {
			out.print(i);
			for (Integer e : phi_w.get(i).keySet()) {
				Fs fs = phi_w.get(i).get(e);
				out.print("\t" + e + " " + fs.alpha + " " + fs.avg_num + " "
						+ fs.lst_upd + " " + fs.upd_val);
			}
			out.println();
		}
		out.println("ENDOFMODEL");
		out.close();
	}

	private void print_XIS_XSI(String fname, FeatureBuilder fb)
			throws FileNotFoundException {
		PrintStream out = new PrintStream(fname);
		for (int i = 0; i < fb.FSIS_size(); ++i)
			out.println(i + "\t" + fb.FSIS(i));
		out.close();
	}

	public void dump(FileWriter tlog) throws IOException {
		int i = 0;
		for (Fs e : phi_s) {
			tlog.write("phi_s[" + i + "]=" + e.dump() + "\n");
			++i;
		}

		i = 0;
		for (Fs e : phi_f) {
			tlog.write("phi_f[" + i + "]=" + e.dump() + "\n");
			++i;
		}

		i = 0;
		for (Fs[] le : phi_b) {
			int j = 0;
			for (Fs e : le) {
				tlog.write("phi_b[" + i + "," + j + "]=" + e.dump() + "\n");
				++j;
			}
			++i;
		}

		// size is k but only some elements used (?)
		int p = 0;
		for (HashMap<Integer, Fs> e : phi_w) {
			Integer sk[] = e.keySet().toArray(new Integer[1]);

			Arrays.sort(sk);
			for (Integer kk : sk) {
				if (kk != null)
					tlog.write("phi_w[" + p + "]=" + e.get(kk) + "\n");
			}
			++p;
		}

		// we are non using active_phi but initilizing it
		for (HashSet<Integer> ap : active_phi_w) {
			Integer sk[] = ap.toArray(new Integer[1]);
			Arrays.sort(sk);
			for (Integer kk : sk) {
				if (kk != null)
					tlog.write("active_phi_w  = " + k + "\n");
			}

		}

		tlog.close();
	}

}
