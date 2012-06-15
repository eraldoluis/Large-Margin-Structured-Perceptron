package br.pucrio.inf.learn.structlearning.discriminative.application.dpgs;

import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileWriter;
import java.io.IOException;
import java.io.Writer;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Set;
import java.util.TreeSet;

import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;
import org.json.JSONTokener;
import org.json.JSONWriter;

import br.pucrio.inf.learn.structlearning.discriminative.application.coreference.CorefColumnDataset;
import br.pucrio.inf.learn.structlearning.discriminative.application.dp.Feature;
import br.pucrio.inf.learn.structlearning.discriminative.application.dp.FeatureTemplate;
import br.pucrio.inf.learn.structlearning.discriminative.application.dp.SimpleFeatureTemplate;
import br.pucrio.inf.learn.structlearning.discriminative.application.dp.data.DPColumnDataset;
import br.pucrio.inf.learn.structlearning.discriminative.application.sequence.AveragedParameter;
import br.pucrio.inf.learn.structlearning.discriminative.data.Dataset;
import br.pucrio.inf.learn.structlearning.discriminative.data.ExampleInput;
import br.pucrio.inf.learn.structlearning.discriminative.data.ExampleOutput;
import br.pucrio.inf.learn.structlearning.discriminative.data.encoding.FeatureEncoding;
import br.pucrio.inf.learn.structlearning.discriminative.task.Model;

/**
 * Represent a dependecy parsing model with gradparent and modifiers paramenters
 * by means of a set of templates that conjoing basic features within the input
 * structure.
 * 
 * @author eraldo
 * 
 */
public class DPGSModel implements Model {

	/**
	 * Special root node.
	 */
	protected int root;

	/**
	 * Weight for each generated feature code (model parameters).
	 */
	protected Map<Integer, AveragedParameter> parameters;

	/**
	 * Set of parameters that have been updated in the current iteration. It is
	 * used by the averaged perceptron.
	 */
	protected Set<AveragedParameter> updatedParameters;

	/**
	 * Grandparent templates that comprise three parameters: head token,
	 * modifier token and head of the head token (grandparent of modifier
	 * token).
	 */
	protected FeatureTemplate[] grandparentTemplates;

	/**
	 * Siblings templates for modifiers on the left side of the head token.
	 * These templates comprise three parameters: head token, modifier token (on
	 * the left side of the head token) and the closest modifier token before
	 * the modifier token. The first sibling token is always START and the last
	 * is END. Both of them are represented by index N, where N is the number of
	 * tokens in the sentence.
	 */
	protected FeatureTemplate[] leftSiblingsTemplates;

	/**
	 * Siblings templates for modifiers on the right side of the head token.
	 * These templates comprise three parameters: head token, modifier token (on
	 * the right side of the head token) and the closest modifier token before
	 * the modifier token. The first sibling token is always START and the last
	 * is END. Both of them are represented by index N, where N is the number of
	 * tokens in the sentence.
	 */
	protected FeatureTemplate[] rightSiblingsTemplates;

	/**
	 * Create a new model with the given root node.
	 * 
	 * @param root
	 *            index of the special node that is to be considered as the
	 *            fixed root node that is always chosen by the prediction
	 *            algorithm.
	 */
	public DPGSModel(int root) {
		this.root = root;
		this.parameters = new HashMap<Integer, AveragedParameter>();
		this.updatedParameters = new HashSet<AveragedParameter>();
	}

	/**
	 * Load a model from the given file and using the encodings in the given
	 * dataset. Usually, the loaded model will later be applied in this dataset.
	 * The dataset encodins can be even empty and then they will be filled with
	 * features from the loaded model.
	 * 
	 * @param fileName
	 * @param dataset
	 * @throws JSONException
	 * @throws IOException
	 */
	public DPGSModel(String fileName, CorefColumnDataset dataset)
			throws JSONException, IOException {
		this.updatedParameters = null;
		this.parameters = new HashMap<Integer, AveragedParameter>();

		// Model file input stream.
		FileInputStream fis = new FileInputStream(fileName);

		// Load JSON model object.
		JSONObject jModel = new JSONObject(new JSONTokener(fis));

		// Set dataset templates.
		FeatureTemplate[][] templatesAllLevels = loadTemplatesFromJSON(jModel,
				dataset);
		dataset.setTemplates(templatesAllLevels);

		// Set model parameters.
		loadParametersFromJSON(jModel, dataset);

		// Close model file input stream.
		fis.close();

		// Get root value.
		this.root = jModel.getInt("root");
	}

	/**
	 * Copy constructor.
	 * 
	 * @param other
	 * @throws CloneNotSupportedException
	 */
	@SuppressWarnings("unchecked")
	protected DPGSModel(DPGSModel other) throws CloneNotSupportedException {
		// Root node.
		this.root = other.root;

		// Shallow-copy parameters map.
		this.parameters = (HashMap<Integer, AveragedParameter>) ((HashMap<Integer, AveragedParameter>) other.parameters)
				.clone();

		// Clone each map value.
		for (Entry<Integer, AveragedParameter> entry : parameters.entrySet())
			entry.setValue(entry.getValue().clone());

		// Updated parameters and features are NOT copied.
		updatedParameters = new TreeSet<AveragedParameter>();
	}

	/**
	 * Load model parameters from the given JSON model object
	 * <code>jModel</code>.
	 * 
	 * @param jModel
	 * @param dataset
	 * @throws JSONException
	 */
	protected void loadParametersFromJSON(JSONObject jModel,
			CorefColumnDataset dataset) throws JSONException {
		// Encodings.
		FeatureEncoding<String> basicEncoding = dataset.getFeatureEncoding();
		FeatureEncoding<Feature> explicitEncoding = dataset
				.getExplicitEncoding();
		// JSON array of parameters.
		JSONArray jParams = jModel.getJSONArray("parameters");
		int numParams = jParams.length();
		for (int idxParam = 0; idxParam < numParams; ++idxParam) {
			/*
			 * JSON array that represents a complete parameter: its template
			 * index, its feature values and its weight.
			 */
			JSONArray jParam = jParams.getJSONArray(idxParam);
			// Template index.
			int idxTpl = jParam.getInt(0);
			// Copy basic features values.
			JSONArray jValues = jParam.getJSONArray(1);
			int[] values = new int[jValues.length()];
			for (int idxVal = 0; idxVal < values.length; ++idxVal)
				values[idxVal] = basicEncoding.put(jValues.getString(idxVal));
			// Create a feature object and encode it.
			Feature ftr = new Feature(idxTpl, values);
			int code = explicitEncoding.put(ftr);
			// Put the new feature weight in the parameters.
			parameters.put(code, new AveragedParameter(jParam.getDouble(2)));
		}
	}

	/**
	 * Load templates from the given JSON model object <code>jModel</code>.
	 * 
	 * @param jModel
	 * @param dataset
	 * @return
	 * @throws JSONException
	 */
	protected FeatureTemplate[][] loadTemplatesFromJSON(JSONObject jModel,
			CorefColumnDataset dataset) throws JSONException {
		// Get template set.
		JSONArray jTemplatesAllLevels = jModel.getJSONArray("templates");
		FeatureTemplate[][] templatesAllLevels = new FeatureTemplate[jTemplatesAllLevels
				.length()][];
		for (int level = 0; level < templatesAllLevels.length; ++level) {
			JSONArray jTemplates = jTemplatesAllLevels.getJSONArray(level);
			FeatureTemplate[] templates = new FeatureTemplate[jTemplates
					.length()];
			for (int idxTpl = 0; idxTpl < templates.length; ++idxTpl) {
				JSONArray jTemplate = jTemplates.getJSONArray(idxTpl);
				int[] features = new int[jTemplate.length()];
				for (int idxFtr = 0; idxFtr < features.length; ++idxFtr)
					features[idxFtr] = dataset.getFeatureIndex(jTemplate
							.getString(idxFtr));
				SimpleFeatureTemplate tpl = new SimpleFeatureTemplate(idxTpl,
						features);
				templates[idxTpl] = tpl;
			}
			templatesAllLevels[level] = templates;
		}
		return templatesAllLevels;
	}

	/**
	 * Return the parameters map.
	 * 
	 * @return
	 */
	public Map<Integer, AveragedParameter> getParameters() {
		return parameters;
	}

	/**
	 * Return the sum of the scores of the given list of features.
	 * 
	 * @param features
	 * @return
	 */
	public double getFeatureListScore(int[] features) {
		if (features == null)
			return Double.NaN;
		double score = 0d;
		for (int code : features) {
			AveragedParameter param = parameters.get(code);
			if (param != null)
				score += param.get();
		}
		return score;
	}

	/**
	 * Return the score of the grandparent factor specified by the given
	 * parameters.
	 * 
	 * @param input
	 * @param idxHead
	 * @param idxModifier
	 * @param idxGrandparent
	 * @return
	 */
	public double getGrandparentFactorScore(DPGSInput input, int idxHead,
			int idxModifier, int idxGrandparent) {
		return getFeatureListScore(input.getGrandParentFeatures(idxHead,
				idxModifier, idxGrandparent));
	}

	/**
	 * Return the score of the modifiers factor specified by the given
	 * parameters.
	 * 
	 * @param input
	 * @param idxHead
	 * @param idxModifier
	 * @param idxSibling
	 * @return
	 */
	public double getSiblingsFactorScore(DPGSInput input, int idxHead,
			int idxModifier, int idxSibling) {
		return getFeatureListScore(input.getSiblingsFeatures(idxHead,
				idxModifier, idxSibling));
	}

	@Override
	public double update(ExampleInput input, ExampleOutput outputCorrect,
			ExampleOutput outputPredicted, double learningRate) {
		return update((DPGSInput) input, (DPGSOutput) outputCorrect,
				(DPGSOutput) outputPredicted, learningRate);
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
	protected double update(DPGSInput input, DPGSOutput outputCorrect,
			DPGSOutput outputPredicted, double learningRate) {
		// Per-edge loss value for this example.
		double loss = 0d;
		int numTkns = input.size();
		for (int idxHead = 0; idxHead < numTkns; ++idxHead) {
			// Correct and predicted grandparent heads.
			int correctGrandparent = outputCorrect.getHead(idxHead);
			int predictedGrandparent = outputPredicted.getGrandparent(idxHead);

			/*
			 * Correct and predicted previous modifier. The numTkns index (i.e.,
			 * the not-in-range last token) is the special index to indicate
			 * START and END symbols.
			 */
			int correctPreviousModifier = numTkns;
			int predictedPreviousModifier = numTkns;
			for (int idxModifier = 0; idxModifier < numTkns; ++idxModifier) {
				/*
				 * Is this modifier included in the correct or in the predicted
				 * structures for the current head.
				 */
				boolean isCorrectModifier = (outputCorrect.getHead(idxModifier) == idxHead);
				boolean isPredictedModifier = outputPredicted.isModifier(
						idxHead, idxModifier);

				if (!isCorrectModifier && !isPredictedModifier)
					/*
					 * Modifier token is included in neither the correct
					 * structure nor the predicted structure. Thus, skip it.
					 */
					continue;

				if (isCorrectModifier != isPredictedModifier) {
					if (isCorrectModifier) {
						/*
						 * Current modifier is correct but the predicted
						 * structure does not set it as a modifier of the
						 * current head (false negative). Thus, increment the
						 * weight of both (grandparent and siblings) correct,
						 * but missed, factors.
						 */
						updateSiblingsFactorParams(input, idxHead, idxModifier,
								correctPreviousModifier, learningRate);
						updateGrandparentFactorParams(input, idxHead,
								idxModifier, correctGrandparent, learningRate);
					} else {
						/*
						 * Current modifier is not correct but the predicted
						 * structure does set it as a modifier of the current
						 * head (false positive). Thus, decrement the weight of
						 * both (grandparent and siblings) incorrectly predicted
						 * factors.
						 */
						updateSiblingsFactorParams(input, idxHead, idxModifier,
								predictedPreviousModifier, -learningRate);
						updateGrandparentFactorParams(input, idxHead,
								idxModifier, predictedGrandparent,
								-learningRate);
					}
				} else {
					/*
					 * The current modifier has been correctly predicted for the
					 * current head. Then, addtionally check the previous
					 * modifier and the grandparent head.
					 */

					if (correctPreviousModifier != predictedPreviousModifier) {
						/*
						 * Modifier is correctly predited but previous modifier
						 * is NOT. Thus, the corresponding correct siblings
						 * factor is missing (false negative) and the predicted
						 * one is incorrectly predicted (false positive).
						 */
						updateSiblingsFactorParams(input, idxHead, idxModifier,
								correctPreviousModifier, learningRate);
						updateSiblingsFactorParams(input, idxHead, idxModifier,
								predictedPreviousModifier, -learningRate);
					}

					if (correctGrandparent != predictedGrandparent) {
						/*
						 * Predicted modifier is correct but grandparent head is
						 * NOT. Thus, the corresponding correct grandparent
						 * factor is missing (false negative) and the predicted
						 * one is incorrectly predicted (false positive).
						 */
						updateGrandparentFactorParams(input, idxHead,
								idxModifier, correctGrandparent, learningRate);
						updateGrandparentFactorParams(input, idxHead,
								idxModifier, predictedGrandparent,
								-learningRate);
					}
				}

				// Update previous modifiers.
				if (isCorrectModifier)
					correctPreviousModifier = idxModifier;
				if (isPredictedModifier)
					predictedPreviousModifier = idxModifier;
			}

			if (correctPreviousModifier != predictedPreviousModifier) {
				/*
				 * The last modifiers of correct and predicted structures are
				 * different. Thus, update the factors from the last modifiers
				 * to the special END symbol (numTkns index).
				 */
				updateSiblingsFactorParams(input, idxHead, numTkns,
						correctPreviousModifier, learningRate);
				updateSiblingsFactorParams(input, idxHead, numTkns,
						predictedPreviousModifier, -learningRate);
			}
		}

		return loss;
	}

	/**
	 * Update all feature parameters in the given grandparent factor of the
	 * given input.
	 * 
	 * @param input
	 * @param idxHead
	 * @param idxModifier
	 * @param idxGrandparent
	 * @param learnRate
	 */
	protected void updateGrandparentFactorParams(DPGSInput input, int idxHead,
			int idxModifier, int idxGrandparent, double learnRate) {
		int[] ftrs = input.getGrandParentFeatures(idxHead, idxModifier,
				idxGrandparent);
		for (int code : ftrs)
			updateFeatureParam(code, learnRate);
	}

	/**
	 * Update all feature parameters in the given grandparent factor of the
	 * given input.
	 * 
	 * @param input
	 * @param idxHead
	 * @param idxModifier
	 * @param idxSibling
	 * @param learnRate
	 */
	protected void updateSiblingsFactorParams(DPGSInput input, int idxHead,
			int idxModifier, int idxSibling, double learnRate) {
		int[] ftrs = input
				.getSiblingsFeatures(idxHead, idxModifier, idxSibling);
		for (int code : ftrs)
			updateFeatureParam(code, learnRate);
	}

	/**
	 * Update the weight of the given parameter with the given value (sum this
	 * value to the given parameter weight). If this parameter has not been
	 * instantiated yet, then instantiate it and initialize its weight to the
	 * given value.
	 * 
	 * @param code
	 *            is the parameter (feature) code to be updated.
	 * @param value
	 *            is the value to be added to the parameter weight.
	 */
	protected void updateFeatureParam(int code, double value) {
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
	public DPGSModel clone() throws CloneNotSupportedException {
		return new DPGSModel(this);
	}

	/**
	 * Return the number of non-zero parameters.
	 * 
	 * @return
	 */
	public int getNumberOfNonZeroParameters() {
		return parameters.size();
	}

	@Override
	public void save(String fileName, Dataset dataset) throws IOException,
			FileNotFoundException {
		FileWriter fw = new FileWriter(fileName);
		save(fw, (DPColumnDataset) dataset);
		fw.close();
	}

	/**
	 * Save this model in the given <code>FileWriter</code> object.
	 * 
	 * @param w
	 * @param dataset
	 * @throws IOException
	 */
	public void save(Writer w, DPColumnDataset dataset) throws IOException {
		FeatureEncoding<String> basicEncoding = dataset.getFeatureEncoding();
		FeatureEncoding<Feature> explicitEncoding = dataset
				.getExplicitEncoding();
		try {
			// JSON objects writer.
			JSONWriter jw = new JSONWriter(w);

			// Model object.
			jw.object();

			// Root value.
			jw.key("root").value(root);

			// Templates array.
			jw.key("templates");
			jw.array();
			FeatureTemplate[][] templatesAllLevels = dataset.getTemplates();
			for (FeatureTemplate[] templates : templatesAllLevels) {
				// Templates array of the current level.
				jw.array();
				for (FeatureTemplate template : templates) {
					// Features array of the current template.
					jw.array();
					for (int idxFtr : template.getFeatures())
						jw.value(dataset.getFeatureLabel(idxFtr));
					// End of features array of the current template.
					jw.endArray();
				}
				// End of templates array of the current level.
				jw.endArray();
			}
			// End of templates array.
			jw.endArray();

			// Parameters array.
			jw.key("parameters");
			jw.array();
			for (Entry<Integer, AveragedParameter> entry : parameters
					.entrySet()) {
				// Explicit features array:
				// [template_index, [feature_values_array], weight].
				jw.array();
				Feature ftr = explicitEncoding.getValueByCode(entry.getKey());
				jw.value(ftr.getTemplateIndex());
				// Feature values array.
				jw.array();
				for (int code : ftr.getValues())
					jw.value(basicEncoding.getValueByCode(code));
				// End of feature values array.
				jw.endArray();
				// Parameter weight.
				jw.value(entry.getValue().get());
				// End of explicit features array.
				jw.endArray();
			}
			// End of parameters array.
			jw.endArray();

			// End of model object.
			jw.endObject();

		} catch (JSONException e) {
			throw new IOException("JSON error", e);
		}
	}

	/**
	 * Sum the parameters of the given model in this model. The given model
	 * parameters are weighted by the given weight.
	 * 
	 * @param model
	 * @param weight
	 */
	public void sumModel(DPGSModel model, double weight) {
		for (Entry<Integer, AveragedParameter> entry : model.parameters
				.entrySet()) {
			int code = entry.getKey();
			double val = entry.getValue().get();
			AveragedParameter param = parameters.get(code);
			if (param == null) {
				param = new AveragedParameter();
				parameters.put(code, param);
			}
			param.increment(val * weight);
		}
	}
}
