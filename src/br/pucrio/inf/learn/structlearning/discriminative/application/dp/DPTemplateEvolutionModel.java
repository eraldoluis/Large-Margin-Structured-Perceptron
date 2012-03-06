package br.pucrio.inf.learn.structlearning.discriminative.application.dp;

import java.io.PrintStream;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedList;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Set;
import java.util.TreeSet;

import sun.reflect.generics.reflectiveObjects.NotImplementedException;
import br.pucrio.inf.learn.structlearning.discriminative.application.dp.data.DPEdgeCorpus;
import br.pucrio.inf.learn.structlearning.discriminative.application.dp.data.DPInput;
import br.pucrio.inf.learn.structlearning.discriminative.application.dp.data.DPOutput;
import br.pucrio.inf.learn.structlearning.discriminative.application.sequence.AveragedParameter;
import br.pucrio.inf.learn.structlearning.discriminative.data.ExampleInput;
import br.pucrio.inf.learn.structlearning.discriminative.data.ExampleOutput;
import br.pucrio.inf.learn.structlearning.discriminative.data.encoding.FeatureEncoding;
import br.pucrio.inf.learn.structlearning.discriminative.data.encoding.MapEncoding;

/**
 * Represent a dependecy parsing model (head-dependent edge parameters) by means
 * of a set of templates that conjoing basic features within the input
 * structure.
 * 
 * In this version, templates are partitioned and each partition is used once at
 * a time. For each partition, some learning iterations are performed
 * considering only the features from this template partition. Then, the current
 * weights for these features are fixed and the corresponding accumulated
 * weights for each edge is stored for efficiency matter and the next template
 * partition is used for the next learning iterations.
 * 
 * 
 * @author eraldo
 * 
 */
public class DPTemplateEvolutionModel implements DPModel {

	/**
	 * Number of template partitions.
	 */
	private int numberOfPartitions;

	/**
	 * Current partition.
	 */
	private int currentPartition;

	/**
	 * Feature templates organized into partitions.
	 */
	private final FeatureTemplate[][] templates;

	/**
	 * Training corpus.
	 */
	private DPEdgeCorpus corpus;

	/**
	 * Encode feature objects as integer codes. A feature object is the
	 * instantiation of a template.
	 */
	private MapEncoding<Feature> encoding;

	/**
	 * Weight for each feature code (model parameters).
	 */
	private Map<Integer, AveragedParameter> parameters;

	/**
	 * Set of parameters that have been updated in the current iteration.
	 */
	private Set<AveragedParameter> updatedParameters;

	/**
	 * Features for the current partition.
	 */
	@SuppressWarnings("rawtypes")
	private LinkedList[][][] activeFeatures;

	/**
	 * Accumulated weight for each edge in the training corpus.
	 */
	private double[][][] fixedWeights;

	/**
	 * Create a new model with the given template partitions.
	 * 
	 * @param templates
	 *            Partitioned templates. This object is not cloned and, thus,
	 *            must be untouched while it is used by this model.
	 * 
	 */
	public DPTemplateEvolutionModel(FeatureTemplate[][] templates) {
		this.numberOfPartitions = templates.length;
		this.currentPartition = -1;
		this.templates = templates;
		encoding = new MapEncoding<Feature>();
		parameters = new HashMap<Integer, AveragedParameter>();
		updatedParameters = new HashSet<AveragedParameter>();
	}

	/**
	 * Copy constructor.
	 * 
	 * @param other
	 * @throws CloneNotSupportedException
	 */
	@SuppressWarnings("unchecked")
	protected DPTemplateEvolutionModel(DPTemplateEvolutionModel other)
			throws CloneNotSupportedException {
		// Templates are shallow copied.
		this.templates = other.templates;

		// Encoding is shallow copied.
		encoding = other.encoding;

		// Shallow-copy parameters map.
		this.parameters = (HashMap<Integer, AveragedParameter>) ((HashMap<Integer, AveragedParameter>) other.parameters)
				.clone();
		// Clone each map value.
		for (Entry<Integer, AveragedParameter> entry : parameters.entrySet())
			entry.setValue(entry.getValue().clone());

		// Updated parameters and features are NOT copied.
		updatedParameters = new TreeSet<AveragedParameter>();

		// Shallow-copy (explicit) active features and fixed weights.
		activeFeatures = other.activeFeatures;
		fixedWeights = other.fixedWeights;
	}

	/**
	 * Initialize the internal data structures.
	 * 
	 * @param corpus
	 */
	public void init(DPEdgeCorpus corpus) {
		this.corpus = corpus;

		// Allocate active features lists and fixed weights matrix.
		DPInput[] inputs = corpus.getInputs();
		int numExs = inputs.length;
		activeFeatures = new LinkedList[numExs][][];
		fixedWeights = new double[numExs][][];
		for (int idxEx = 0; idxEx < numExs; ++idxEx) {
			DPInput input = inputs[idxEx];
			int numTkns = input.getNumberOfTokens();
			activeFeatures[idxEx] = new LinkedList[numTkns][numTkns];
			fixedWeights[idxEx] = new double[numTkns][numTkns];
			for (int idxHead = 0; idxHead < numTkns; ++idxHead)
				for (int idxDep = 0; idxDep < numTkns; ++idxDep)
					if (input.getFeatureCodes(idxHead, idxDep) != null)
						activeFeatures[idxEx][idxHead][idxDep] = new LinkedList<Integer>();
		}

		// Generate feature lists for the first template partition.
		currentPartition = 0;
		generateFeatures();
	}

	/**
	 * Store the accumulated weight of each edge for the current template
	 * partition and generate the features for the next partition.
	 * 
	 * @return the next partition.
	 */
	public int nextPartition() {
		// Input structures.
		DPInput[] inputs = corpus.getInputs();
		int numExs = inputs.length;

		// Accumulate current partition feature weights.
		for (int idxEx = 0; idxEx < numExs; ++idxEx) {
			DPInput input = inputs[idxEx];
			int numTkns = input.getNumberOfTokens();
			for (int idxHead = 0; idxHead < numTkns; ++idxHead) {
				for (int idxDep = 0; idxDep < numTkns; ++idxDep) {
					double score = getEdgeScoreFromCurrentFeatures(input,
							idxHead, idxDep);
					if (!Double.isNaN(score))
						fixedWeights[idxEx][idxHead][idxDep] += score;
				}
			}
		}

		// Go to next partition and generate new features.
		++currentPartition;
		if (currentPartition < numberOfPartitions)
			generateFeatures();
		return currentPartition;
	}

	/**
	 * Generate features for the current template partition.
	 */
	protected void generateFeatures() {
		// Current template evolution partition.
		FeatureTemplate[] tpls = templates[currentPartition];
		DPInput[] inputs = corpus.getInputs();
		int numExs = inputs.length;
		for (int idxEx = 0; idxEx < numExs; ++idxEx) {
			DPInput input = inputs[idxEx];
			int numTkns = input.getNumberOfTokens();
			for (int head = 0; head < numTkns; ++head) {
				for (int dependent = 0; dependent < numTkns; ++dependent) {
					@SuppressWarnings("unchecked")
					LinkedList<Integer> ftrs = activeFeatures[idxEx][head][dependent];
					if (ftrs == null)
						// Skip non-existent edge.
						continue;
					ftrs.clear();

					/*
					 * Instantiate edge features and add them to active features
					 * list.
					 */
					for (int idxTpl = 0; idxTpl < tpls.length; ++idxTpl) {
						int globalIdxTpl = currentPartition
								* numberOfPartitions + idxTpl;
						FeatureTemplate tpl = tpls[idxTpl];
						// Get temporary feature instance.
						Feature ftr = tpl.getInstance(input, head, dependent,
								globalIdxTpl);
						// Lookup the feature in the encoding.
						int code = encoding.getCodeByValue(ftr);
						/*
						 * Instantiate a new feature, if it is not present in
						 * the encoding.
						 */
						if (code == FeatureEncoding.UNSEEN_VALUE_CODE)
							code = encoding.put(tpl.newInstance(input, head,
									dependent, globalIdxTpl));
						// Add feature code to active features list.
						ftrs.add(code);
					}
				}
			}

			// Progess report.
			if ((idxEx + 1) % 100 == 0) {
				System.out.print('.');
				System.out.flush();
			}
		}

		System.out.println();
		System.out.flush();
	}

	/**
	 * Return an edge weight based only on the current features in
	 * <code>activeFeatures</code> list.
	 * 
	 * @param input
	 * @param idxHead
	 * @param idxDependent
	 * @return
	 */
	protected double getEdgeScoreFromCurrentFeatures(DPInput input,
			int idxHead, int idxDependent) {
		int idxEx = input.getTrainingIndex();
		@SuppressWarnings("unchecked")
		LinkedList<Integer> activeFtrs = activeFeatures[idxEx][idxHead][idxDependent];

		// Check edge existence.
		if (activeFtrs == null)
			return Double.NaN;

		double score = 0d;
		for (int code : activeFtrs) {
			AveragedParameter param = parameters.get(code);
			if (param != null)
				score += param.get();
		}

		return score;
	}

	@Override
	public double getEdgeScore(DPInput input, int idxHead, int idxDependent) {
		int idxEx = input.getTrainingIndex();
		double score = getEdgeScoreFromCurrentFeatures(input, idxHead,
				idxDependent);
		return fixedWeights[idxEx][idxHead][idxDependent] + score;
	}

	@Override
	public double update(ExampleInput input, ExampleOutput outputCorrect,
			ExampleOutput outputPredicted, double learningRate) {
		return update((DPInput) input, (DPOutput) outputCorrect,
				(DPOutput) outputPredicted, learningRate);
	}

	/**
	 * Update this model using the differences between the correct output and
	 * the predicted output, both given as arguments.
	 * 
	 * @param input
	 * @param outputCorrect
	 * @param outputPredicted
	 * @param learningRate
	 * @return
	 */
	@SuppressWarnings("unchecked")
	private double update(DPInput input, DPOutput outputCorrect,
			DPOutput outputPredicted, double learningRate) {
		/*
		 * The root token (zero) must always be ignored during the inference,
		 * thus it has to be always correctly classified.
		 */
		assert outputCorrect.getHead(0) == outputPredicted.getHead(0);

		int idxEx = input.getTrainingIndex();

		// Per-token loss value for this example.
		double loss = 0d;
		for (int idxTkn = 1; idxTkn < input.getNumberOfTokens(); ++idxTkn) {
			// Correct head token.
			int idxCorrectHead = outputCorrect.getHead(idxTkn);

			// Predicted head token.
			int idxPredictedHead = outputPredicted.getHead(idxTkn);

			// Skip. Correctly predicted head.
			if (idxCorrectHead == idxPredictedHead)
				continue;

			if (idxCorrectHead == -1)
				/*
				 * Skip tokens with missing CORRECT edge (this is due to prune
				 * preprocessing).
				 */
				continue;

			/*
			 * Misclassified head for this token. Update parameters.
			 */

			// Generate missed correct features.
			for (int code : (LinkedList<Integer>) activeFeatures[idxEx][idxCorrectHead][idxTkn])
				updateFeatureParam(code, learningRate);
			for (int code : (LinkedList<Integer>) activeFeatures[idxEx][idxPredictedHead][idxTkn])
				updateFeatureParam(code, -learningRate);

			// Increment (per-token) loss value.
			loss += 1d;
		}

		return loss;
	}

	/**
	 * Recover the parameter associated with the given feature.
	 * 
	 * If the parameter has not been initialized yet, then create it. If the
	 * inverted index is activated and the parameter has not been initialized
	 * yet, then update the active features lists for each edge where the
	 * feature occurs.
	 * 
	 * @param ftr
	 * @param value
	 * @return
	 */
	private void updateFeatureParam(int code, double value) {
		AveragedParameter param = parameters.get(code);
		if (param == null) {
			// Create a new parameter.
			param = new AveragedParameter();
			parameters.put(code, param);
		}

		// Update parameter value.
		param.update(value);

		// Keep track of updated parameter within this example.
		updatedParameters.add(param);
	}

	@Override
	public void sumUpdates(int iteration) {
		for (AveragedParameter parm : updatedParameters)
			parm.sum(iteration);
		updatedParameters.clear();
	}

	@Override
	public void average(int numberOfIterations) {
		for (AveragedParameter parm : parameters.values())
			parm.average(numberOfIterations);
	}

	@Override
	public DPTemplateEvolutionModel clone() throws CloneNotSupportedException {
		return new DPTemplateEvolutionModel(this);
	}

	@Override
	public void save(PrintStream ps, FeatureEncoding<String> featureEncoding,
			FeatureEncoding<String> stateEncoding) {
		throw new NotImplementedException();
	}

	@Override
	public int getNonZeroParameters() {
		return parameters.size();
	}

}
