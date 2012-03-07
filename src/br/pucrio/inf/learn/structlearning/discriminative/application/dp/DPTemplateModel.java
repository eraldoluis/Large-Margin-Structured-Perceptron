package br.pucrio.inf.learn.structlearning.discriminative.application.dp;

import java.io.PrintStream;
import java.util.Collection;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedList;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Set;
import java.util.TreeSet;

import sun.reflect.generics.reflectiveObjects.NotImplementedException;
import br.pucrio.inf.learn.structlearning.discriminative.application.dp.InvertedIndex.Edge;
import br.pucrio.inf.learn.structlearning.discriminative.application.dp.data.DPColumnDataset;
import br.pucrio.inf.learn.structlearning.discriminative.application.dp.data.DPInput;
import br.pucrio.inf.learn.structlearning.discriminative.application.dp.data.DPOutput;
import br.pucrio.inf.learn.structlearning.discriminative.application.sequence.AveragedParameter;
import br.pucrio.inf.learn.structlearning.discriminative.data.ExampleInput;
import br.pucrio.inf.learn.structlearning.discriminative.data.ExampleOutput;
import br.pucrio.inf.learn.structlearning.discriminative.data.encoding.FeatureEncoding;

/**
 * Represent a dependecy parsing model (head-dependent edge parameters) by means
 * of a set of templates that conjoing basic features within the input
 * structure.
 * 
 * @author eraldo
 * 
 */
public class DPTemplateModel implements DPModel {

	/**
	 * Weight for each feature code. A feature code represents a template
	 * instantiation.
	 */
	protected Map<Feature, AveragedParameter> parameters;

	/**
	 * Set of parameters that have been updated in the current iteration.
	 */
	protected Set<AveragedParameter> updatedParameters;

	/**
	 * Corpus inverted index.
	 */
	protected InvertedIndex index;

	/**
	 * Missed features to be updated (increment weight).
	 */
	protected HashSet<Feature> updateMissedFeatures;

	/**
	 * Wrong recovered features to be updated (decrement weight).
	 */
	protected HashSet<Feature> updateWrongFeatures;

	/**
	 * Create a new model with the given template set.
	 */
	public DPTemplateModel() {
		parameters = new HashMap<Feature, AveragedParameter>();
		updatedParameters = new HashSet<AveragedParameter>();
		updateMissedFeatures = new HashSet<Feature>();
		updateWrongFeatures = new HashSet<Feature>();
	}

	/**
	 * Copy constructor.
	 * 
	 * @param other
	 * @throws CloneNotSupportedException
	 */
	@SuppressWarnings("unchecked")
	protected DPTemplateModel(DPTemplateModel other)
			throws CloneNotSupportedException {
		// Shallow-copy parameters map and then clone each parameter.
		this.parameters = (HashMap<Feature, AveragedParameter>) ((HashMap<Feature, AveragedParameter>) other.parameters)
				.clone();
		for (Entry<Feature, AveragedParameter> entry : parameters.entrySet())
			entry.setValue(entry.getValue().clone());

		// Updated parameters and features are NOT copied.
		updatedParameters = new TreeSet<AveragedParameter>();
		updateMissedFeatures = new HashSet<Feature>();
		updateWrongFeatures = new HashSet<Feature>();
	}

	// /**
	// * Initialize internal data structures to use inverted index or explicit
	// * features lists that can be optionally activated in the given corpus.
	// *
	// * @param corpus
	// */
	// public void init(DPColumnDataset corpus) {
	// index = corpus.getInvertedIndex();
	// if (index != null) {
	// /*
	// * Use inverted index. The active features matrix
	// * (<code>activeFeatures</code>) stores the derived features whose
	// * value is different from zero. The inverted index is used to
	// * efficiently instantiate derived features on all examples when
	// * their value is changed from zero.
	// */
	// DPInput[] inputs = corpus.getInputs();
	// int numExs = corpus.getNumberOfExamples();
	// activeFeatures = new LinkedList[numExs][][];
	// for (int idxEx = 0; idxEx < numExs; ++idxEx) {
	// int lenEx = inputs[idxEx].getNumberOfTokens();
	// activeFeatures[idxEx] = new LinkedList[lenEx][lenEx];
	// if ((idxEx + 1) % 100 == 0) {
	// System.out.print(".");
	// System.out.flush();
	// }
	// }
	// System.out.println();
	// }
	// }

	@SuppressWarnings("unchecked")
	@Override
	public double getEdgeScore(DPInput input, int idxHead, int idxDependent) {
		// Check edge existence.
		if (input.getFeatures(idxHead, idxDependent) == null)
			return Double.NaN;

		double score = 0d;

		if (activeFeatures == null) {
			if (input.getFeatures(idxHead, idxDependent) != null) {
				// Generate all features from templates.
				for (int idxTemplate = 0; idxTemplate < templates.length; ++idxTemplate) {
					AveragedParameter param = parameters
							.get(templates[idxTemplate].getInstance(input,
									idxHead, idxDependent));
					if (param == null)
						// Feature not instantiated yet, thus its value is zero.
						continue;
					// Accumulate the parameter in the edge score.
					score += param.get();
				}
			}
		} else {
			// Use active features only.
			LinkedList<Feature> activeFtrs = activeFeatures[input
					.getTrainingIndex()][idxHead][idxDependent];
			if (activeFtrs != null) {
				for (Feature ftr : activeFtrs) {
					AveragedParameter param = parameters.get(ftr);
					if (param != null)
						// Accumulate the parameter in the edge score.
						score += param.get();
				}
			}
		}

		return score;
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
	private double update(DPInput input, DPOutput outputCorrect,
			DPOutput outputPredicted, double learningRate) {
		/*
		 * The root token (zero) must always be ignored during the inference,
		 * thus it has to be always correctly classified.
		 */
		assert outputCorrect.getHead(0) == outputPredicted.getHead(0);

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
			for (int idxTemplate = 0; idxTemplate < templates.length; ++idxTemplate)
				updateMissedFeatures.add(templates[idxTemplate].newInstance(
						input, idxCorrectHead, idxTkn));

			// Generate wrongly misrecovered features.
			for (int idxTemplate = 0; idxTemplate < templates.length; ++idxTemplate) {
				// Instantiate feature.
				Feature ftr = templates[idxTemplate].newInstance(input,
						idxPredictedHead, idxTkn);
				if (!updateMissedFeatures.remove(ftr))
					/*
					 * Include feature in the wrongly recovered list only if it
					 * is not present in the missed feature list.
					 */
					updateWrongFeatures.add(ftr);
			}

			// Update missed features.
			for (Feature ftr : updateMissedFeatures)
				updateFeatureParam(ftr, learningRate);

			// Update mis-recovered features.
			for (Feature ftr : updateWrongFeatures)
				updateFeatureParam(ftr, -learningRate);

			// Clear update feature sets.
			updateMissedFeatures.clear();
			updateWrongFeatures.clear();

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
	@SuppressWarnings("unchecked")
	private void updateFeatureParam(Feature ftr, double value) {
		AveragedParameter param = parameters.get(ftr);
		if (param == null) {
			// Create a new parameter.
			param = new AveragedParameter();
			parameters.put(ftr, param);

			if (index != null) {
				// Update active features.
				Collection<Edge> edges = index.getExamplesWithFeatures(
						templates[ftr.getTemplateIndex()].getFeatures(),
						ftr.getValues());
				if (edges != null) {
					for (Edge e : edges) {
						LinkedList<Feature> activeFtrs = activeFeatures[e.example][e.head][e.dependent];
						if (activeFtrs == null) {
							activeFtrs = new LinkedList<Feature>();
							activeFeatures[e.example][e.head][e.dependent] = activeFtrs;
						}
						activeFtrs.add(ftr);
					}
				}
			}
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
	public DPTemplateModel clone() throws CloneNotSupportedException {
		return new DPTemplateModel(this);
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
