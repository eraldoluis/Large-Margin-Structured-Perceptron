package br.pucrio.inf.learn.structlearning.data;

import java.util.Map.Entry;
import java.util.TreeMap;

/**
 * Sparse representation of a feature-weight vector.
 * 
 * @author eraldof
 * 
 */
public class FeatureVector {

	private TreeMap<Feature, Double> featureWeights;

	public FeatureVector() {
		featureWeights = new TreeMap<Feature, Double>();
	}

	public FeatureVector(FeatureVector copy) {
		featureWeights = new TreeMap<Feature, Double>(copy.featureWeights);
	}

	public FeatureVector increment(FeatureVector inc) {
		// TODO
		return this;
	}

	public FeatureVector decrement(FeatureVector dec) {
		// TODO
		return this;
	}

	public FeatureVector difference(FeatureVector diff) {
		// TODO
		return this;
	}

	public FeatureVector scale(double learningRate) {
		for (Entry<Feature, Double> entry : featureWeights.entrySet())
			entry.setValue(entry.getValue() * learningRate);
		return this;
	}

	public void increment(Feature feature, double val) {
		if (val == 0d)
			return;
		Double weight = featureWeights.get(feature);
		if (weight == null)
			weight = new Double(val);
		else
			weight = weight + val;
		featureWeights.put(feature, weight);
	}

	public double get(Feature feature) {
		Double weight = featureWeights.get(feature);
		if (weight == null)
			return 0d;
		return weight.doubleValue();
	}

}
