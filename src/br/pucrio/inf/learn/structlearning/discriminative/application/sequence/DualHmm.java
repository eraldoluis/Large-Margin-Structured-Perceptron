package br.pucrio.inf.learn.structlearning.discriminative.application.sequence;

import java.util.Arrays;
import java.util.HashMap;
import java.util.Iterator;
import java.util.Map.Entry;
import java.util.Set;
import java.util.TreeMap;
import java.util.TreeSet;

import sun.reflect.generics.reflectiveObjects.NotImplementedException;
import br.pucrio.inf.learn.structlearning.discriminative.application.sequence.data.SequenceInput;
import br.pucrio.inf.learn.structlearning.discriminative.application.sequence.data.SequenceOutput;
import br.pucrio.inf.learn.structlearning.discriminative.data.ExampleInputArray;
import br.pucrio.inf.learn.structlearning.discriminative.data.ExampleOutput;
import br.pucrio.inf.learn.structlearning.discriminative.task.DualModel;
import br.pucrio.inf.learn.structlearning.discriminative.task.Inference;
import br.pucrio.inf.learn.util.HashCodeUtil;

/**
 * Represent HMM emission parameters through a dual representation, i.e., store
 * counts for each sequence token that was misclassified at some point of the
 * learning algorithm. Additionally, store primal transition parameters.
 * 
 * @author eraldof
 * 
 */
public class DualHmm extends Hmm implements DualModel {

	/**
	 * Number of possible states.
	 */
	private int numberOfStates;

	/**
	 * Power of the polynomial kernel.
	 */
	private int exponent;

	/**
	 * Base input patterns used to represent the weights. These are the
	 * candidates for support vector.
	 */
	private final ExampleInputArray inputs;

	/**
	 * Correct output patterns for each base input pattern.
	 */
	private final SequenceOutput[] outputs;

	/**
	 * Transition parameters. They are directly represented in the primal (no
	 * kernel function for them).
	 */
	private AveragedParameter[][] transitions;

	/**
	 * Initial state parameters.
	 */
	private AveragedParameter[] initialStates;

	/**
	 * Dual representation of the emission parameters. For each misclassified
	 * token in the input patterns, there is an array of counters (alphas) which
	 * include a counter for each possible state.
	 */
	private TreeMap<Integer, TreeMap<Integer, TreeMap<Integer, AveragedParameter>>> dualEmissionVariables;

	/**
	 * Set of parameters updated in the current iteration. It is used to speedup
	 * the averaged (or voted) perceptron.
	 */
	private Set<AveragedParameter> updatedParameters;

	/**
	 * Indicate when the distillation process is on going.
	 */
	private boolean distillationOnGoing;

	/**
	 * Create a dual HMM for the given input/output patterns with the given
	 * number of states.
	 * 
	 * @param inputs
	 * @param outputs
	 * @param numberOfStates
	 * @param exponent
	 */
	public DualHmm(final ExampleInputArray inputs,
			final SequenceOutput[] outputs, int numberOfStates, int exponent) {
		// Input/output patterns.
		this.inputs = inputs;
		this.outputs = outputs;

		// Total number of states.
		this.numberOfStates = numberOfStates;

		// Polynomial kernel exponent.
		this.exponent = exponent;

		// Allocate data structures.
		transitions = new AveragedParameter[numberOfStates][numberOfStates];
		initialStates = new AveragedParameter[numberOfStates];
		dualEmissionVariables = new TreeMap<Integer, TreeMap<Integer, TreeMap<Integer, AveragedParameter>>>();

		for (int state1 = 0; state1 < numberOfStates; ++state1) {
			initialStates[state1] = new AveragedParameter();
			for (int state2 = 0; state2 < numberOfStates; ++state2)
				transitions[state1][state2] = new AveragedParameter();
		}

		// Updated parameters in each iteration.
		this.updatedParameters = new TreeSet<AveragedParameter>();
	}

	/**
	 * Partial constructor used only locally to implement clone method.
	 * 
	 * @param inputs
	 * @param outputs
	 */
	protected DualHmm(final ExampleInputArray inputs,
			final SequenceOutput[] outputs) {
		// Input/output patterns.
		this.inputs = inputs;
		this.outputs = outputs;

		// Updated parameters in each iteration.
		this.updatedParameters = new TreeSet<AveragedParameter>();
	}

	/**
	 * The sub-classes must implement this to ease some use cases (e.g.,
	 * evaluating itermediate models during the execution of a training
	 * algorithm).
	 */
	@SuppressWarnings("unchecked")
	@Override
	public DualHmm clone() throws CloneNotSupportedException {
		DualHmm copy = new DualHmm(inputs, outputs);
		copy.numberOfStates = numberOfStates;
		copy.exponent = exponent;
		copy.initialStates = initialStates.clone();
		copy.transitions = transitions.clone();

		// Deep copy of the initial state and transition arrays.
		for (int state1 = 0; state1 < numberOfStates; ++state1) {
			copy.initialStates[state1] = initialStates[state1].clone();
			copy.transitions[state1] = transitions[state1].clone();
			for (int state2 = 0; state2 < numberOfStates; ++state2)
				copy.transitions[state1][state2] = transitions[state1][state2]
						.clone();
		}

		// Shallow copy of the dual variables map.
		copy.dualEmissionVariables = (TreeMap<Integer, TreeMap<Integer, TreeMap<Integer, AveragedParameter>>>) ((TreeMap<Integer, TreeMap<Integer, TreeMap<Integer, AveragedParameter>>>) dualEmissionVariables)
				.clone();

		// Deep copy of the map *values* (the keys do not need to be cloned).
		for (Entry<Integer, TreeMap<Integer, TreeMap<Integer, AveragedParameter>>> clonedEntrySequence : copy.dualEmissionVariables
				.entrySet()) {
			// Original sequence.
			TreeMap<Integer, TreeMap<Integer, AveragedParameter>> sequence = clonedEntrySequence
					.getValue();
			// Shallowly-cloned sequence.
			TreeMap<Integer, TreeMap<Integer, AveragedParameter>> clonedSequence = (TreeMap<Integer, TreeMap<Integer, AveragedParameter>>) sequence
					.clone();
			// Set the map entry value of the cloned sequence.
			clonedEntrySequence.setValue(clonedSequence);

			// Deeply clone the sequence.
			for (Entry<Integer, TreeMap<Integer, AveragedParameter>> clonedEntryToken : clonedSequence
					.entrySet()) {
				// Original token.
				TreeMap<Integer, AveragedParameter> token = clonedEntryToken
						.getValue();
				// Shallowly-cloned token.
				TreeMap<Integer, AveragedParameter> clonedToken = (TreeMap<Integer, AveragedParameter>) token
						.clone();
				// Set the map entry value of the cloned token.
				clonedEntryToken.setValue(clonedToken);

				// Deeply clone the token.
				for (Entry<Integer, AveragedParameter> clonedEntryVariable : clonedToken
						.entrySet()) {
					// Original variable.
					AveragedParameter alpha = clonedEntryVariable.getValue();
					// Clone the alpha variable.
					clonedEntryVariable.setValue(alpha.clone());
				}
			}
		}

		return copy;
	}

	@Override
	public int getNumberOfExamplesWithSupportVector() {
		return dualEmissionVariables.size();
	}

	@Override
	public int getNumberOfSupportVectors() {
		int count = 0;
		for (TreeMap<Integer, TreeMap<Integer, AveragedParameter>> sequence : dualEmissionVariables
				.values())
			count += sequence.size();
		return count;
	}

	@Override
	public int getNumberOfStates() {
		return numberOfStates;
	}

	@Override
	public double getInitialStateParameter(int state) {
		return initialStates[state].get();
	}

	@Override
	public double getTransitionParameter(int fromState, int toState) {
		return transitions[fromState][toState].get();
	}

	@Override
	public void setInitialStateParameter(int state, double value) {
		initialStates[state].set(value);
	}

	@Override
	public void setTransitionParameter(int fromState, int toState, double value) {
		transitions[fromState][toState].set(value);
	}

	HashMap<DualEmissionKeyPair, Double> kernelCache = new HashMap<DualHmm.DualEmissionKeyPair, Double>();

	/**
	 * Distil the current set of support vectors. For each sequence that
	 * contains some support vector, classify it using the given margin (
	 * <code>lossWeight</code>) and disconsidering its support vectors in the
	 * model. Then, remove all correctly classified support vectors. If all
	 * support vectors in a sequence are correctly classified, then remove the
	 * whole sequence from the model.
	 * 
	 * @param inference
	 *            algorithm to perform loss-augmented inferences.
	 * @param lossWeight
	 *            per-token margin value.
	 * @param outputsCache
	 *            array with allocated output structures used to store the
	 *            predicted structures given by the
	 */
	@Override
	public void distill(Inference inference, double lossWeight,
			ExampleOutput[] outputsCache) {
		// Activate ditilation.
		distillationOnGoing = true;

		// Iterator over the sequences.
		Iterator<Entry<Integer, TreeMap<Integer, TreeMap<Integer, AveragedParameter>>>> itVars = dualEmissionVariables
				.entrySet().iterator();
		while (itVars.hasNext()) {
			// Current sequence (map) entry.
			Entry<Integer, TreeMap<Integer, TreeMap<Integer, AveragedParameter>>> entrySequence = itVars
					.next();

			/*
			 * Index of the sequence and its corresponding input and outputs
			 * structures.
			 */
			int idxSequence = entrySequence.getKey();
			
			inputs.load(new int[idxSequence]);
			
			SequenceInput input = (SequenceInput) inputs.get(idxSequence);
			SequenceOutput output = outputs[idxSequence];
			SequenceOutput predicted = (SequenceOutput) outputsCache[idxSequence];

			/*
			 * Infer the current sequence tags ignoring the support vectors
			 * associated with the current sequence.
			 */
			inference.lossAugmentedInference(this, input, output, predicted,
					lossWeight);

			/*
			 * Iterate over the support vectors in the current sequence and
			 * remove the correctly classified ones.
			 * 
			 * TODO maybe it needs to include the misclassified ones.
			 */
			Iterator<Entry<Integer, TreeMap<Integer, AveragedParameter>>> itSequence = entrySequence
					.getValue().entrySet().iterator();
			while (itSequence.hasNext()) {
				Entry<Integer, TreeMap<Integer, AveragedParameter>> entryToken = itSequence
						.next();
				int idxToken = entryToken.getKey();
				if (output.getLabel(idxToken) == predicted.getLabel(idxToken)) {
					// Clean kernel function cache.
					for (Entry<Integer, TreeMap<Integer, TreeMap<Integer, AveragedParameter>>> _entrySequence : dualEmissionVariables
							.entrySet())
						for (Entry<Integer, TreeMap<Integer, AveragedParameter>> _entryToken : _entrySequence
								.getValue().entrySet())
							kernelCache.remove(new DualEmissionKeyPair(
									idxSequence, idxToken, _entrySequence
											.getKey(), _entryToken.getKey()));

					// Remove correctly classified token from model.
					itSequence.remove();
				}
			}

			if (entrySequence.getValue().size() == 0)
				// Remove correctly classified sequences.
				itVars.remove();
		}

		// Deactivate distilation.
		distillationOnGoing = false;
	}

	/**
	 * Current example temporary cache. It is used to avoid recalculating kernel
	 * function values that have been calculated and then are added as support
	 * vectors.
	 */
	double[][][] kernelCacheCurrentExample;

	@Override
	public void getTokenEmissionWeights(SequenceInput input, int idxTknTrain,
			double[] weights) {
		// Clear weights array.
		Arrays.fill(weights, 0d);

		/*
		 * Index of the given sequence within the training dataset, if it is
		 * part of one.
		 */
		int idxSeqTrain = input.getTrainingIndex();

		if (!distillationOnGoing) {
			if (idxTknTrain == 0)
				kernelCacheCurrentExample = new double[input.size()][][];
			kernelCacheCurrentExample[idxTknTrain] = new double[dualEmissionVariables
					.size()][];
		}

		int _idxSeqSV = 0;
		// For each sequence.
		for (Entry<Integer, TreeMap<Integer, TreeMap<Integer, AveragedParameter>>> entrySequence : dualEmissionVariables
				.entrySet()) {

			// Current sequence index.
			int idxSeqSV = entrySequence.getKey();

			TreeMap<Integer, TreeMap<Integer, AveragedParameter>> sequence = entrySequence
					.getValue();

			int _idxTknSV = 0;
			if (!distillationOnGoing)
				kernelCacheCurrentExample[idxTknTrain][_idxSeqSV] = new double[sequence
						.size()];

			// For each token within the current sequence.
			for (Entry<Integer, TreeMap<Integer, AveragedParameter>> entryToken : sequence
					.entrySet()) {

				// Currect token index.
				int idxTknSV = entryToken.getKey();

				// Calculate the kernel function value.
				double k;
				if (distillationOnGoing) {
					if (idxSeqTrain == idxSeqSV && idxTknTrain == idxTknSV) {
						/*
						 * Do not use the kernel function value between a
						 * support vector and itself, in order to calculate the
						 * prediction when removing this support vector.
						 */
						++_idxTknSV;
						continue;
					} else {
						/*
						 * Kernel function values between two support vectors
						 * are cached.
						 */
						k = kernelCache.get(new DualEmissionKeyPair(idxSeqSV,
								idxTknSV, idxSeqTrain, idxTknTrain));
					}
				} else {
					/*
					 * Evaluate the kernel function between the training example
					 * token and the current support vector.
					 */
					inputs.load(new int[idxSeqSV]);
					
					k = kernel((SequenceInput) inputs.get(idxSeqSV), idxTknSV, input, idxTknTrain);

					/*
					 * Store the kernel function value in the current example
					 * temporary cache.
					 */
					kernelCacheCurrentExample[idxTknTrain][_idxSeqSV][_idxTknSV] = k;
				}

				/*
				 * Sum the kernel function values weighted by the alpha
				 * counters.
				 */
				for (Entry<Integer, AveragedParameter> entryAlpha : entryToken
						.getValue().entrySet()) {
					int state = entryAlpha.getKey();
					double alpha = entryAlpha.getValue().get();
					weights[state] += alpha * k;
				}

				++_idxTknSV;
			}

			++_idxSeqSV;
		}
	}

	/**
	 * Polynomial kernel function for two given tokens. The feature indexes
	 * within each token must be sorted in increasing order.
	 * 
	 * @param input1
	 * @param token1
	 * @param input2
	 * @param token2
	 * @return
	 */
	protected double kernel(SequenceInput input1, int token1,
			SequenceInput input2, int token2) {
		int idxFtr1 = 0;
		int idxFtr2 = 0;
		int numFtrs1 = input1.getNumberOfInputFeatures(token1);
		int numFtrs2 = input2.getNumberOfInputFeatures(token2);
		double dotProd = 0d;
		while (idxFtr1 < numFtrs1 && idxFtr2 < numFtrs2) {
			int ftrVal1 = input1.getFeature(token1, idxFtr1);
			int ftrVal2 = input2.getFeature(token2, idxFtr2);
			if (ftrVal1 == ftrVal2) {
				dotProd += input1.getFeatureWeight(token1, idxFtr1)
						* input2.getFeatureWeight(token2, idxFtr2);
				++idxFtr1;
				++idxFtr2;
			} else if (ftrVal1 > ftrVal2)
				++idxFtr2;
			else
				++idxFtr1;
		}

		switch (exponent) {
		case 1:
			// Linear kernel.
			return dotProd;
		case 2:
			// Quadratic kernel.
			return dotProd * dotProd;
		case 3:
			// Cubic kernel.
			return dotProd * dotProd * dotProd;
		case 4:
			// Quartic kernel.
			return dotProd * dotProd * dotProd * dotProd;
		}

		return 0d;
	}

	@Override
	protected void updateInitialStateParameter(int state, double value) {
		initialStates[state].update(value);
		updatedParameters.add(initialStates[state]);
	}

	@Override
	protected void updateTransitionParameter(int fromState, int toState,
			double value) {
		transitions[fromState][toState].update(value);
		updatedParameters.add(transitions[fromState][toState]);
	}

	/**
	 * Update emission dual parameters (alphas) for the sequence and token by
	 * comparing with the given predicted output.
	 * 
	 * @param sequenceId
	 * @param idxToken
	 * @param labelCorrect
	 * @param labelPredicted
	 * @param learnRate
	 */
	protected void updateEmissionParameters(int sequenceId, int idxToken,
			int labelCorrect, int labelPredicted, double learnRate) {

		/*
		 * Store kernel function values in the cache. These values have been
		 * stored along previous getTokenEmissinoWeights calls.
		 * 
		 * TODO if (sequenceId, idxToken) is already an SV, it does not need to
		 * be included. On the other hand, the new SVs are added here one by
		 * one. Once the first one is added, the order within the
		 * <code>dualEmissionVariables</code> mapping is modified and does not
		 * respect the order within the kernel function temporary cache
		 * <code>kernelCacheCurrentExample</code>. This should not happen.
		 */

		// Index of SV sequence in the current example temporary cache.
		int _idxSeqSV = 0;

		// For each sequence.
		TreeMap<Integer, TreeMap<Integer, AveragedParameter>> sequence;
		for (Entry<Integer, TreeMap<Integer, TreeMap<Integer, AveragedParameter>>> entrySequence : dualEmissionVariables
				.entrySet()) {

			// Current sequence index.
			int idxSeqSV = entrySequence.getKey();

			sequence = entrySequence.getValue();

			// For each token within the current sequence SV.
			int _idxTknSV = 0;
			for (Entry<Integer, TreeMap<Integer, AveragedParameter>> entryToken : sequence
					.entrySet()) {

				// Currect token index.
				int idxTknSV = entryToken.getKey();

				// Store kernel function value.
				kernelCache
						.put(new DualEmissionKeyPair(idxSeqSV, idxTknSV,
								sequenceId, idxToken),
								kernelCacheCurrentExample[idxToken][_idxSeqSV][_idxTknSV]);

				++_idxTknSV;
			}

			++_idxSeqSV;
		}

		/*
		 * Store kernel function value between the new support vector and
		 * itself.
		 */
		
		inputs.load(new int [sequenceId]);
		
		kernelCache.put(
				new DualEmissionKeyPair(sequenceId, idxToken, sequenceId,
						idxToken),
				kernel((SequenceInput) inputs.get(sequenceId), idxToken, (SequenceInput) inputs.get(sequenceId),
						idxToken));

		// Sequence variables.
		sequence = dualEmissionVariables.get(sequenceId);
		if (sequence == null) {
			sequence = new TreeMap<Integer, TreeMap<Integer, AveragedParameter>>();
			dualEmissionVariables.put(sequenceId, sequence);
		}

		// Token variables.
		TreeMap<Integer, AveragedParameter> token = sequence.get(idxToken);
		if (token == null) {
			token = new TreeMap<Integer, AveragedParameter>();
			sequence.put(idxToken, token);
		}

		// Correct label parameter.
		AveragedParameter parmLabelCorrect = token.get(labelCorrect);
		if (parmLabelCorrect == null) {
			parmLabelCorrect = new AveragedParameter();
			token.put(labelCorrect, parmLabelCorrect);
		}

		// Predicted label parameter.
		AveragedParameter parmLabelPredicted = token.get(labelPredicted);
		if (parmLabelPredicted == null) {
			parmLabelPredicted = new AveragedParameter();
			token.put(labelPredicted, parmLabelPredicted);
		}

		// Update parameters.
		parmLabelCorrect.update(learnRate);
		parmLabelPredicted.update(-learnRate);

		// Keep track of the updated parameters.
		updatedParameters.add(parmLabelCorrect);
		updatedParameters.add(parmLabelPredicted);
	}

	@Override
	public double update(int sequenceId, ExampleOutput outputReference,
			ExampleOutput outputPredicted, double learnRate) {
		// Just cast the structures to sequences.
		return update(sequenceId, (SequenceOutput) outputReference,
				(SequenceOutput) outputPredicted, learnRate);
	}

	/**
	 * Update this model using the input structure in the given index and the
	 * given two output structures, the correct (reference) output and the
	 * predicted one.
	 * 
	 * @param sequenceId
	 * @param outputReference
	 * @param outputPredicted
	 * @param learnRate
	 * @return the loss value for this example.
	 */
	public double update(int sequenceId, SequenceOutput outputReference,
			SequenceOutput outputPredicted, double learnRate) {
		// The loss value.
		double loss = 0d;

		// Skip empty examples.
		if (outputPredicted.size() <= 0)
			return 0d;

		// Input structure.
		inputs.load(new int[sequenceId]);
		
		SequenceInput input = (SequenceInput) inputs.get(sequenceId);
		// Correct output structure.
		SequenceOutput outputCorrect = outputReference;

		// First token.
		int labelCorrect = outputCorrect.getLabel(0);
		int labelPredicted = outputPredicted.getLabel(0);
		if (labelCorrect != labelPredicted) {
			// Initial state parameters.
			updateInitialStateParameter(labelCorrect, learnRate);
			updateInitialStateParameter(labelPredicted, -learnRate);
			// Emission parameters.
			updateEmissionParameters(sequenceId, 0, labelCorrect,
					labelPredicted, learnRate);
			// Update loss (per-token).
			loss += 1;
		}

		int prevLabelCorrect = labelCorrect;
		int prevLabelPredicted = labelPredicted;
		for (int tkn = 1; tkn < input.size(); ++tkn) {
			labelCorrect = outputCorrect.getLabel(tkn);
			labelPredicted = outputPredicted.getLabel(tkn);
			if (labelCorrect != labelPredicted) {
				// Emission parameters.
				updateEmissionParameters(sequenceId, tkn, labelCorrect,
						labelPredicted, learnRate);
				// Transition parameters.
				updateTransitionParameter(prevLabelCorrect, labelCorrect,
						learnRate);
				updateTransitionParameter(prevLabelPredicted, labelPredicted,
						-learnRate);
				// Update loss (per-token).
				loss += 1;
			} else if (prevLabelCorrect != prevLabelPredicted) {
				// Transition parameters.
				updateTransitionParameter(prevLabelCorrect, labelCorrect,
						learnRate);
				updateTransitionParameter(prevLabelPredicted, labelPredicted,
						-learnRate);
			}

			prevLabelCorrect = labelCorrect;
			prevLabelPredicted = labelPredicted;
		}

		return loss;
	}

	@Override
	public void sumUpdates(int iteration) {
		for (AveragedParameter param : updatedParameters)
			param.sum(iteration);
		updatedParameters.clear();
	}

	@Override
	public void average(int numberOfIterations) {
		// Average all transition and initial state parameters of this model.
		for (int state1 = 0; state1 < numberOfStates; ++state1) {
			initialStates[state1].average(numberOfIterations);
			for (int state2 = 0; state2 < numberOfStates; ++state2)
				transitions[state1][state2].average(numberOfIterations);
		}

		// Average all emission parameters within this model.
		for (Entry<Integer, TreeMap<Integer, TreeMap<Integer, AveragedParameter>>> entrySequence : dualEmissionVariables
				.entrySet()) {
			for (Entry<Integer, TreeMap<Integer, AveragedParameter>> entryToken : entrySequence
					.getValue().entrySet()) {
				for (Entry<Integer, AveragedParameter> entryAlpha : entryToken
						.getValue().entrySet()) {
					AveragedParameter alpha = entryAlpha.getValue();
					alpha.average(numberOfIterations);
				}
			}
		}
	}

	@Override
	public int getNumberOfSymbols() {
		throw new NotImplementedException();
	}

	@Override
	public void setEmissionParameter(int state, int symbol, double value) {
		throw new NotImplementedException();
	}

	@Override
	public double getEmissionParameter(int state, int symbol) {
		throw new NotImplementedException();
	}

	@Override
	protected void updateEmissionParameters(SequenceInput input, int token,
			int state, double learningRate) {
		throw new NotImplementedException();
	}

	/**
	 * Key used to store the alpha values within a sparse data structure (hash
	 * table). The key represents a token within a specific example.
	 * 
	 * @author eraldo
	 * 
	 */
	private static final class DualEmissionKey implements
			Comparable<DualEmissionKey> {

		/**
		 * The index within the array of input sequences in the
		 * <code>DualHmm</code> object.
		 */
		private final int sequenceId;

		/**
		 * The token index within the sequence.
		 */
		private final int token;

		/**
		 * Create a new key representing the given values.
		 * 
		 * @param sequenceId
		 * @param token
		 */
		public DualEmissionKey(int sequenceId, int token) {
			this.sequenceId = sequenceId;
			this.token = token;
		}

		@Override
		public int hashCode() {
			return HashCodeUtil.hash(HashCodeUtil.hash(sequenceId), token);
		}

		@Override
		public boolean equals(Object obj) {
			if (obj.getClass() != getClass())
				return false;
			DualEmissionKey other = (DualEmissionKey) obj;
			return sequenceId == other.sequenceId && token == other.token;
		}

		@Override
		public DualEmissionKey clone() throws CloneNotSupportedException {
			return new DualEmissionKey(sequenceId, token);
		}

		@Override
		public int compareTo(DualEmissionKey o) {
			if (sequenceId < o.sequenceId)
				return -1;
			if (sequenceId > o.sequenceId)
				return 1;

			// Sequence IDs are equal.
			if (token < o.token)
				return -1;
			if (token > o.token)
				return 1;

			// Pairs (sequence ID, token) are equal.
			return 0;
		}
	}

	/**
	 * Pair of <code>DualEmissionKey</code>s used as the key in the cache of
	 * kernel function values.
	 * 
	 * @author eraldo
	 * 
	 */
	private static final class DualEmissionKeyPair {

		/**
		 * Key with smaller index.
		 */
		private final DualEmissionKey key1;

		/**
		 * Key with bigger index.
		 */
		private final DualEmissionKey key2;

		/**
		 * Create a key pair from the given pair of keys.
		 * 
		 * @param key1
		 * @param key2
		 */
		public DualEmissionKeyPair(DualEmissionKey key1, DualEmissionKey key2) {
			// The first key must always be the smallest one.
			if (key1.compareTo(key2) <= 0) {
				this.key1 = key1;
				this.key2 = key2;
			} else {
				this.key1 = key2;
				this.key2 = key1;
			}
		}

		/**
		 * Create a key pair from the given integers.
		 * 
		 * @param idxSeq1
		 * @param idxTkn1
		 * @param idxSeq2
		 * @param idxTkn2
		 */
		public DualEmissionKeyPair(int idxSeq1, int idxTkn1, int idxSeq2,
				int idxTkn2) {
			this(new DualEmissionKey(idxSeq1, idxTkn1), new DualEmissionKey(
					idxSeq2, idxTkn2));
		}

		@Override
		public int hashCode() {
			return HashCodeUtil.hash(key1.hashCode(), key2.hashCode());
		}

		@Override
		public boolean equals(Object obj) {
			if (obj == null)
				return false;
			if (getClass() != obj.getClass())
				return false;

			// Same type.
			DualEmissionKeyPair pair = (DualEmissionKeyPair) obj;
			return key1.equals(pair.key1) && key2.equals(pair.key2);
		}

		@Override
		public DualEmissionKeyPair clone() throws CloneNotSupportedException {
			return new DualEmissionKeyPair(key1, key2);
		}
	}
}
