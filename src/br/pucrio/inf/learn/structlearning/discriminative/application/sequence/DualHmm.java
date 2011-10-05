package br.pucrio.inf.learn.structlearning.discriminative.application.sequence;

import java.util.Arrays;
import java.util.Comparator;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.Map;
import java.util.PriorityQueue;
import java.util.Set;
import java.util.TreeSet;
import java.util.Map.Entry;

import sun.reflect.generics.reflectiveObjects.NotImplementedException;
import br.pucrio.inf.learn.structlearning.discriminative.application.sequence.data.SequenceInput;
import br.pucrio.inf.learn.structlearning.discriminative.application.sequence.data.SequenceOutput;
import br.pucrio.inf.learn.structlearning.discriminative.data.ExampleOutput;
import br.pucrio.inf.learn.structlearning.discriminative.task.DualModel;
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
	private final SequenceInput[] inputs;

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
	private Map<DualEmissionKey, AveragedParameter[]> dualEmissionVariables;

	/**
	 * Set of parameters updated in the current iteration. It is used to speedup
	 * the averaged (or voted) perceptron.
	 */
	private Set<AveragedParameter> updatedParameters;

	/**
	 * Kernel function cache that stores previous calculated kernel function
	 * values.
	 */
	private Map<DualEmissionKeyPair, Double> kernelFunctionCache;

	/**
	 * Create a dual HMM for the given input/output patterns with the given
	 * number of states.
	 * 
	 * @param inputs
	 * @param outputs
	 * @param numberOfStates
	 * @param exponent
	 */
	public DualHmm(final SequenceInput[] inputs,
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
		dualEmissionVariables = new HashMap<DualHmm.DualEmissionKey, AveragedParameter[]>();

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
	protected DualHmm(final SequenceInput[] inputs,
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
		copy.dualEmissionVariables = (Map<DualEmissionKey, AveragedParameter[]>) ((HashMap<DualEmissionKey, AveragedParameter[]>) dualEmissionVariables)
				.clone();

		// Deep copy of the map *values* (the keys are not cloned).
		for (Entry<DualEmissionKey, AveragedParameter[]> entry : copy.dualEmissionVariables
				.entrySet()) {
			AveragedParameter[] clonedArray = entry.getValue().clone();
			entry.setValue(clonedArray);
			for (int state = 0; state < numberOfStates; ++state)
				clonedArray[state] = clonedArray[state].clone();
		}

		if (kernelFunctionCache != null)
			/*
			 * The copy will use kernel function cache only if this object does
			 * so.
			 */
			copy.kernelFunctionCache = new HashMap<DualHmm.DualEmissionKeyPair, Double>();

		return copy;
	}

	/**
	 * Number of tokens used as support vectors. Do not consider how many states
	 * within each token has weight great than zero though.
	 * 
	 * @return
	 */
	public int getNumberOfSupportVectors() {
		return dualEmissionVariables.size();
	}

	/**
	 * Return the number of possible states (labels) of this model.
	 * 
	 * @return
	 */
	public int getNumberOfStates() {
		return numberOfStates;
	}

	/**
	 * Return the weight associated with the given initial state.
	 * 
	 * @param state
	 * @return
	 */
	public double getInitialStateParameter(int state) {
		return initialStates[state].get();
	}

	/**
	 * Return the weight associated with the transition from the two given
	 * states.
	 * 
	 * @param fromState
	 *            the origin state.
	 * @param toState
	 *            the end state.
	 * @return
	 */
	public double getTransitionParameter(int fromState, int toState) {
		return transitions[fromState][toState].get();
	}

	/**
	 * Set the value (weight) of the initial parameter associated with the given
	 * state.
	 * 
	 * @param state
	 * @param value
	 */
	public void setInitialStateParameter(int state, double value) {
		initialStates[state].set(value);
	}

	/**
	 * Set the value (weight) of the transition parameter associated with the
	 * given pair of states.
	 * 
	 * @param fromState
	 * @param toState
	 * @param value
	 */
	public void setTransitionParameter(int fromState, int toState, double value) {
		transitions[fromState][toState].set(value);
	}

	/**
	 * Activate or deactivate the kernel function cache for this model.
	 * 
	 * @param activate
	 */
	public void setActivateKernelFunctionCache(boolean activate) {
		if (activate) {
			if (kernelFunctionCache == null || kernelFunctionCache.size() > 0)
				kernelFunctionCache = new HashMap<DualHmm.DualEmissionKeyPair, Double>();
		} else
			kernelFunctionCache = null;
	}

	@Override
	public void getTokenEmissionWeights(SequenceInput input, int token,
			double[] weights) {
		// Clear weights array.
		Arrays.fill(weights, 0d);

		/*
		 * Index of the given sequence within the training dataset, if it is
		 * part of one.
		 */
		int trainingIndex = input.getTrainingIndex();

		// Use kernel function cache?
		boolean cache = (kernelFunctionCache != null && trainingIndex >= 0);

		for (Entry<DualEmissionKey, AveragedParameter[]> alphaEntry : dualEmissionVariables
				.entrySet()) {
			DualEmissionKey key = alphaEntry.getKey();

			/*
			 * Evaluate the kernel function between the current support vector
			 * and the given token.
			 */
			double k;
			if (cache) {
				/*
				 * Kernel function cache is activated and the given sequence is
				 * a training sequence. So, query the cache for the desired
				 * value.
				 */
				DualEmissionKeyPair keyPair = new DualEmissionKeyPair(key,
						new DualEmissionKey(trainingIndex, token));
				Double kCached = kernelFunctionCache.get(keyPair);
				if (kCached == null) {
					/*
					 * If the desired value is not present in the cache, we
					 * calculate it and put it in the cache.
					 */
					kCached = kernel(inputs[key.sequenceId], key.token, input,
							token);
					kernelFunctionCache.put(keyPair, kCached);
				}

				k = kCached;
			} else
				k = kernel(inputs[key.sequenceId], key.token, input, token);

			// Calculate the emission weight associated with each state.
			// TODO it can be worthy to represent the list of states sparsely.
			AveragedParameter[] alphas = alphaEntry.getValue();
			for (int state = 0; state < numberOfStates; ++state)
				weights[state] += alphas[state].get() * k;
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

	/**
	 * Add the given value to the initial-state parameter of the mobel.
	 * 
	 * @param state
	 * @param value
	 */
	protected void updateInitialStateParameter(int state, double value) {
		initialStates[state].update(value);
		updatedParameters.add(initialStates[state]);
	}

	/**
	 * Update the specified transition (fromToken, toToken) feature using the
	 * given learning rate.
	 * 
	 * @param fromState
	 * @param toState
	 * @param value
	 */
	protected void updateTransitionParameter(int fromState, int toState,
			double value) {
		transitions[fromState][toState].update(value);
		updatedParameters.add(transitions[fromState][toState]);
	}

	private int maxAlphaEntries = 4000;
	private PriorityQueue<DualEmissionComparableByAlphaEntropy> alphaPriorityQueue = new PriorityQueue<DualEmissionComparableByAlphaEntropy>(
			maxAlphaEntries);

	private static class DualEmissionComparableByAlphaEntropy implements
			Comparable<DualEmissionComparableByAlphaEntropy> {

		private final DualEmissionKey key;

		private double entropy;

		public DualEmissionComparableByAlphaEntropy(DualEmissionKey key) {
			this.key = key;
		}

		public double updateEntropy(AveragedParameter[] alphas) {
			// Normalization factor.
			double sum = 0d;
			for (AveragedParameter alpha : alphas)
				sum += Math.exp(alpha.get());

			// Alphas entropy.
			entropy = 0d;
			for (AveragedParameter alpha : alphas) {
				double p = Math.exp(alpha.get()) / sum;
				entropy -= p * Math.log(p);
			}

			return entropy;
		}

		@Override
		public int compareTo(DualEmissionComparableByAlphaEntropy o) {
			// Inverse order by entropy.
			if (entropy > o.entropy)
				return -1;
			if (entropy < o.entropy)
				return 1;
			return 0;
		}

		@Override
		public boolean equals(Object obj) {
			if (obj == null)
				return false;
			if (getClass() != obj.getClass())
				return false;
			return key.equals(((DualEmissionComparableByAlphaEntropy) obj).key);
		}

	}

	/**
	 * Update emission dual parameters (alphas) for the sequence and token by
	 * comparing with the given predicted output.
	 * 
	 * @param sequenceId
	 * @param token
	 * @param labelCorrect
	 * @param labelPredicted
	 * @param learnRate
	 */
	protected void updateEmissionParameters(int sequenceId, int token,
			int labelCorrect, int labelPredicted, double learnRate) {
		DualEmissionKey key = new DualEmissionKey(sequenceId, token);
		AveragedParameter[] alphas = dualEmissionVariables.get(key);
		DualEmissionComparableByAlphaEntropy entropyKey = new DualEmissionComparableByAlphaEntropy(
				key);
		if (alphas == null) {
			/*
			 * Limit the quantity of active support vectors by removing the
			 * oldest one.
			 */
			if (alphaPriorityQueue.size() == maxAlphaEntries) {
				DualEmissionComparableByAlphaEntropy smallestKey = alphaPriorityQueue
						.poll();
				dualEmissionVariables.remove(smallestKey.key);
				alphaPriorityQueue.remove(smallestKey);
			}

			alphas = new AveragedParameter[numberOfStates];
			for (int state = 0; state < numberOfStates; ++state)
				alphas[state] = new AveragedParameter();
			dualEmissionVariables.put(key, alphas);
		} else
			alphaPriorityQueue.remove(entropyKey);

		alphas[labelCorrect].update(learnRate);
		alphas[labelPredicted].update(-learnRate);
		updatedParameters.add(alphas[labelCorrect]);
		updatedParameters.add(alphas[labelPredicted]);

		entropyKey.updateEntropy(alphas);
		alphaPriorityQueue.add(entropyKey);
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
		SequenceInput input = inputs[sequenceId];
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
		// Deep copy of the initial state and transition arrays.
		for (int state1 = 0; state1 < numberOfStates; ++state1) {
			initialStates[state1].average(numberOfIterations);
			for (int state2 = 0; state2 < numberOfStates; ++state2)
				transitions[state1][state2].average(numberOfIterations);
		}

		// Deep copy of the map *values* (the keys are not cloned).
		for (Entry<DualEmissionKey, AveragedParameter[]> entry : dualEmissionVariables
				.entrySet()) {
			AveragedParameter[] alphas = entry.getValue();
			for (int state = 0; state < numberOfStates; ++state)
				alphas[state].average(numberOfIterations);
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
		 * Create a key from the given pair.
		 * 
		 * @param key1
		 * @param key2
		 */
		public DualEmissionKeyPair(DualEmissionKey key1, DualEmissionKey key2) {
			this.key1 = key1;
			this.key2 = key2;
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
