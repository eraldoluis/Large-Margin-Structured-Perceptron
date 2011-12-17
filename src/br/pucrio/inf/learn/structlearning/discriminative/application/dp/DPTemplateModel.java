package br.pucrio.inf.learn.structlearning.discriminative.application.dp;

import java.io.PrintStream;
import java.util.HashMap;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Set;
import java.util.TreeSet;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;

import sun.reflect.generics.reflectiveObjects.NotImplementedException;
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
 * @author eraldo
 * 
 */
public class DPTemplateModel implements DPModel {

	/**
	 * Logging object.
	 */
	private static final Log LOG = LogFactory.getLog(DPTemplateModel.class);

	/**
	 * Feature templates.
	 */
	private FeatureTemplate[] templates;

	/**
	 * Weight for each feature code. A feature code represents a template
	 * instantiation.
	 */
	private Map<Integer, AveragedParameter> parameters;

	/**
	 * Encode features as integers.
	 */
	private MapEncoding<Feature> encoding;

	/**
	 * Set of parameters that have been updated in the current iteration.
	 */
	private Set<AveragedParameter> updatedWeights;

	/**
	 * Create an empty model.
	 */
	public DPTemplateModel(FeatureTemplate[] templates) {
		this.templates = templates;
		parameters = new HashMap<Integer, AveragedParameter>();
		encoding = new MapEncoding<Feature>();
		updatedWeights = new TreeSet<AveragedParameter>();
	}

	/**
	 * Copy constructor.
	 * 
	 * @param other
	 */
	protected DPTemplateModel(DPTemplateModel other) {
		// Templates are shallow-copied.
		this.templates = other.templates;

		// Parameters.
		this.parameters = new HashMap<Integer, AveragedParameter>();
		try {
			for (Entry<Integer, AveragedParameter> entry : other.parameters
					.entrySet())
				parameters.put(entry.getKey(), entry.getValue().clone());
		} catch (CloneNotSupportedException e) {
			LOG.error("Clone error.", e);
		}

		// Encoding.
		encoding = new MapEncoding<Feature>();
		for (Feature ftr : other.encoding.getValues())
			encoding.put(ftr);

		// Updated weights are NOT copied.
		updatedWeights = new TreeSet<AveragedParameter>();
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
		 * The root token (zero) must always be ignored during the inference and
		 * so its always correctly classified (its head must always be pointing
		 * to itself).
		 */
		assert outputCorrect.getHead(0) == outputPredicted.getHead(0);

		// Per-token loss value for this example.
		double loss = 0d;
		for (int idxTkn = 1; idxTkn < input.getNumberOfTokens(); ++idxTkn) {
			int idxCorrectHead = outputCorrect.getHead(idxTkn);
			int idxPredictedHead = outputPredicted.getHead(idxTkn);
			if (idxCorrectHead == idxPredictedHead)
				// Correctly predicted head.
				continue;

			if (idxCorrectHead == -1)
				/*
				 * Skip tokens with missing CORRECT edge (this is due to prune
				 * preprocessing).
				 */
				continue;

			//
			// Misclassified head.
			//
			// Increment missed edges weights.
			for (int idxTemplate = 0; idxTemplate < templates.length; ++idxTemplate) {
				int code = encoding.put(templates[idxTemplate].instantiate(
						input, idxCorrectHead, idxTkn, idxTemplate));
				AveragedParameter param = parameters.get(code);
				if (param == null) {
					// Create a new parameter.
					param = new AveragedParameter();
					parameters.put(code, param);
				}

				// Update parameter value.
				param.update(learningRate);
				updatedWeights.add(param);
			}

			// Decrement mispredicted edges weights.
			for (int idxTemplate = 0; idxTemplate < templates.length; ++idxTemplate) {
				int code = encoding.put(templates[idxTemplate].instantiate(
						input, idxPredictedHead, idxTkn, idxTemplate));
				AveragedParameter param = parameters.get(code);
				if (param == null) {
					// Create a new parameter.
					param = new AveragedParameter();
					parameters.put(code, param);
				}

				// Update parameter value.
				param.update(learningRate);
				updatedWeights.add(param);
			}

			// Increment (per-token) loss value.
			loss += 1d;
		}

		return loss;
	}

	@Override
	public void sumUpdates(int iteration) {
		for (AveragedParameter parm : updatedWeights)
			parm.sum(iteration);
		updatedWeights.clear();
	}

	@Override
	public void average(int numberOfIterations) {
		for (AveragedParameter parm : parameters.values())
			parm.average(numberOfIterations);
	}

	@Override
	public double getEdgeScore(DPInput input, int idxHead, int idxDependent) {
		// Check edge existence.
		if (input.getFeatureCodes(idxHead, idxDependent) == null)
			return Double.NEGATIVE_INFINITY;

		double score = 0d;
		for (int idxTemplate = 0; idxTemplate < templates.length; ++idxTemplate) {
			int code = encoding.getCodeByValue(templates[idxTemplate]
					.instantiate(input, idxHead, idxDependent, idxTemplate));
			if (code == MapEncoding.UNSEEN_VALUE_CODE)
				// Feature not instantiated yet, thus it is value is zero.
				continue;

			AveragedParameter param = parameters.get(code);
			if (param == null)
				continue;

			// Accumulate the parameter in the edge score.
			score += param.get();
		}

		return score;
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

}
