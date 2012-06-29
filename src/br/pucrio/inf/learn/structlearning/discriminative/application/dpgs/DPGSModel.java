package br.pucrio.inf.learn.structlearning.discriminative.application.dpgs;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Set;
import java.util.TreeSet;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;

import sun.reflect.generics.reflectiveObjects.NotImplementedException;
import br.pucrio.inf.learn.structlearning.discriminative.application.dp.Feature;
import br.pucrio.inf.learn.structlearning.discriminative.application.sequence.AveragedParameter;
import br.pucrio.inf.learn.structlearning.discriminative.data.Dataset;
import br.pucrio.inf.learn.structlearning.discriminative.data.ExampleInput;
import br.pucrio.inf.learn.structlearning.discriminative.data.ExampleOutput;
import br.pucrio.inf.learn.structlearning.discriminative.data.encoding.MapEncoding;
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
	 * Loging object.
	 */
	private static final Log LOG = LogFactory.getLog(DPGSModel.class);

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
	 * Encoding for explicit features, i.e., features created from templates by
	 * conjoining basic features.
	 */
	protected MapEncoding<Feature> explicitEncoding;

	/**
	 * Grandparent templates that comprise three parameters: head token,
	 * modifier token and head of the head token (grandparent of modifier
	 * token).
	 */
	protected DPGSTemplate[] grandparentTemplates;

	/**
	 * Siblings templates for modifiers on the left side of the head token.
	 * These templates comprise three parameters: head token, modifier token (on
	 * the left side of the head token) and the closest modifier token before
	 * the modifier token. The first sibling token is always START and the last
	 * is END. Both of them are represented by index N, where N is the number of
	 * tokens in the sentence.
	 */
	protected DPGSTemplate[] leftSiblingsTemplates;

	/**
	 * Siblings templates for modifiers on the right side of the head token.
	 * These templates comprise three parameters: head token, modifier token (on
	 * the right side of the head token) and the closest modifier token before
	 * the modifier token. The first sibling token is always START and the last
	 * is END. Both of them are represented by index N, where N is the number of
	 * tokens in the sentence.
	 */
	protected DPGSTemplate[] rightSiblingsTemplates;

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
		this.explicitEncoding = new MapEncoding<Feature>();
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
		int numFtrs = features.length;
		for (int idxCode = 0; idxCode < numFtrs; ++idxCode) {
			AveragedParameter param = parameters.get(features[idxCode]);
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
		return getFeatureListScore(input.getGrandparentFeatures(idxHead,
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
		// Per-factor loss value for this example.
		double loss = 0d;

		/*
		 * For each head and modifier, check whether the predicted factor does
		 * not correspond to the correct one and, then, update the current model
		 * properly.
		 */
		int numTkns = input.size();
		for (int idxHead = 0; idxHead < numTkns; ++idxHead) {
			// Correct and predicted grandparent heads.
			int correctGrandparent = outputCorrect.getHead(idxHead);
			int predictedGrandparent = outputPredicted.getGrandparent(idxHead);

			/*
			 * Verifiy grandparent and siblings factors for differences between
			 * correct and predicted factors.
			 * 
			 * We start as previous token with the special 'idxHead' index is
			 * the index to indicate START and END tokens for LEFT modifiers.
			 * For RIGHT modifiers, we use the 'numTkns' index.
			 */
			int correctPreviousModifier = idxHead;
			int predictedPreviousModifier = idxHead;
			for (int idxModifier = 0; idxModifier <= numTkns; ++idxModifier) {
				// Is this token special (START or END).
				boolean isSpecialToken = (idxModifier == idxHead || idxModifier == numTkns);
				/*
				 * Is this modifier included in the correct or in the predicted
				 * structures for the current head or is it a special token.
				 */
				boolean isCorrectModifier = (isSpecialToken || (outputCorrect
						.getHead(idxModifier) == idxHead));
				boolean isPredictedModifier = (isSpecialToken || outputPredicted
						.isModifier(idxHead, idxModifier));

				if (!isCorrectModifier && !isPredictedModifier)
					/*
					 * Modifier token is neither included in the correct
					 * structure nor the predicted structure. Thus, skip it.
					 */
					continue;

				if (isCorrectModifier != isPredictedModifier) {
					// One error.
					loss += 1;

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
						if (correctGrandparent != -1)
							updateGrandparentFactorParams(input, idxHead,
									idxModifier, correctGrandparent,
									learningRate);
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
						if (predictedGrandparent != -1)
							updateGrandparentFactorParams(input, idxHead,
									idxModifier, predictedGrandparent,
									-learningRate);
					}
				} else {
					// Error flag.
					boolean error = false;

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
						error = true;
					}

					if (!isSpecialToken
							&& correctGrandparent != predictedGrandparent) {
						/*
						 * Predicted modifier is correct but grandparent head is
						 * NOT. Thus, the corresponding correct grandparent
						 * factor is missing (false negative) and the predicted
						 * one is incorrectly predicted (false positive).
						 */
						if (correctGrandparent != -1)
							updateGrandparentFactorParams(input, idxHead,
									idxModifier, correctGrandparent,
									learningRate);
						if (predictedGrandparent != -1)
							updateGrandparentFactorParams(input, idxHead,
									idxModifier, predictedGrandparent,
									-learningRate);
						error = true;
					}

					if (error)
						loss += 1;
				}

				if (isCorrectModifier) {
					// Update correct previous modifier.
					if (idxModifier == idxHead)
						/*
						 * Current token (idxToken) is the boundary token
						 * between left and right modifiers. Thus, the previous
						 * modifier for the next iteration is the special START
						 * token for right modifiers, that is 'numTkns'.
						 */
						correctPreviousModifier = numTkns;
					else
						correctPreviousModifier = idxModifier;
				}

				if (isPredictedModifier) {
					// Update predicted previous modifier.
					if (idxModifier == idxHead)
						/*
						 * Current token (idxToken) is the boundary token
						 * between left and right modifiers. Thus, the previous
						 * modifier for the next iteration is the special START
						 * token for right modifiers, that is 'numTkns'.
						 */
						predictedPreviousModifier = numTkns;
					else
						predictedPreviousModifier = idxModifier;
				}
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
		int[] ftrs = input.getGrandparentFeatures(idxHead, idxModifier,
				idxGrandparent);
		if (ftrs == null)
			// Inexistent factor.
			return;
		for (int idxFtr = 0; idxFtr < ftrs.length; ++idxFtr)
			updateFeatureParam(ftrs[idxFtr], learnRate);
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
		if (ftrs == null)
			// Inexistent factor.
			return;
		for (int idxFtr = 0; idxFtr < ftrs.length; ++idxFtr)
			updateFeatureParam(ftrs[idxFtr], learnRate);
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

	/**
	 * Return the explicit feature encoding of this dataset.
	 * 
	 * @return
	 */
	public MapEncoding<Feature> getExplicitFeatureEncoding() {
		return explicitEncoding;
	}

	public DPGSTemplate[] loadSiblingsTemplates(BufferedReader reader,
			DPGSDataset dataset, int type) throws IOException, DPGSException {
		LinkedList<DPGSTemplate> templatesList = new LinkedList<DPGSTemplate>();
		String line = DPGSDataset.skipBlanksAndComments(reader);
		while (line != null) {
			String[] ftrsStr = line.split("[ ]");
			int[] ftrs = new int[ftrsStr.length];
			for (int idx = 0; idx < ftrs.length; ++idx) {
				ftrs[idx] = dataset.getSiblingsFeatureIndex(ftrsStr[idx]);
				if (ftrs[idx] == -1)
					throw new DPGSException(String.format(
							"Feature label %s does not exist", ftrsStr[idx]));
			}
			templatesList.add(new DPSiblingsTemplate(type,
					templatesList.size(), ftrs));
			// Read next line.
			line = DPGSDataset.skipBlanksAndComments(reader);
		}

		// Convert list to array.
		return templatesList.toArray(new DPGSTemplate[0]);
	}

	/**
	 * Load templates from the given reader and, optionally, generate explicit
	 * features.
	 * 
	 * @param reader
	 * @param dataset
	 * @throws IOException
	 * @throws DPGSException
	 */
	public void loadGrandparentTemplates(BufferedReader reader,
			DPGSDataset dataset) throws IOException, DPGSException {
		LinkedList<DPGSTemplate> templatesList = new LinkedList<DPGSTemplate>();
		String line = DPGSDataset.skipBlanksAndComments(reader);
		while (line != null) {
			String[] ftrsStr = line.split("[ ]");
			int[] ftrs = new int[ftrsStr.length];
			for (int idx = 0; idx < ftrs.length; ++idx) {
				ftrs[idx] = dataset.getGrandparentFeatureIndex(ftrsStr[idx]);
				if (ftrs[idx] == -1)
					throw new DPGSException(String.format(
							"Feature label %s does not exist", ftrsStr[idx]));
			}
			templatesList.add(new DPGrandparentTemplate(templatesList.size(),
					ftrs));
			// Read next line.
			line = DPGSDataset.skipBlanksAndComments(reader);
		}

		// Convert list to array.
		grandparentTemplates = templatesList.toArray(new DPGSTemplate[0]);
	}

	public void loadGrandparentTemplates(String templatesFileName,
			DPGSDataset dataset) throws IOException, DPGSException {
		BufferedReader reader = new BufferedReader(new FileReader(
				templatesFileName));
		loadGrandparentTemplates(reader, dataset);
		reader.close();
	}

	public void loadLeftSiblingsTemplates(String templatesFileName,
			DPGSDataset dataset) throws IOException, DPGSException {
		BufferedReader reader = new BufferedReader(new FileReader(
				templatesFileName));
		leftSiblingsTemplates = loadSiblingsTemplates(reader, dataset, 2);
		reader.close();
	}

	public void loadRightSiblingsTemplates(String templatesFileName,
			DPGSDataset dataset) throws IOException, DPGSException {
		BufferedReader reader = new BufferedReader(new FileReader(
				templatesFileName));
		rightSiblingsTemplates = loadSiblingsTemplates(reader, dataset, 3);
		reader.close();
	}

	protected void instantiateSiblingsFeatures(DPGSInput input, int idxHead,
			int idxModifier, int idxPrevModifier) {
		// Skip non-existent edges.
		if (input.getBasicSiblingsFeatures(idxHead, idxModifier,
				idxPrevModifier) == null)
			return;

		// List of generated features for the current factor.
		LinkedList<Integer> ftrs = new LinkedList<Integer>();

		/*
		 * Instantiate edge features and add them to active features list.
		 */
		for (int idxTpl = 0; idxTpl < grandparentTemplates.length; ++idxTpl) {
			DPSiblingsTemplate tpl = (DPSiblingsTemplate) leftSiblingsTemplates[idxTpl];
			try {
				tpl.instantiateSiblingsDerivedFeatures(input, ftrs,
						explicitEncoding, idxHead, idxModifier, idxPrevModifier);
			} catch (CloneNotSupportedException e) {
				LOG.error("Instantiating feature", e);
			}
		}

		// Set feature vector of this input.
		int numFtrs = ftrs.size();
		int[] ftrVals = new int[numFtrs];
		Iterator<Integer> itFtrVals = ftrs.iterator();
		for (int idxFtr = 0; idxFtr < numFtrs; ++idxFtr)
			ftrVals[idxFtr] = itFtrVals.next();
		input.setSiblingsFeatures(idxHead, idxModifier, idxPrevModifier,
				ftrVals);
	}

	public void generateFeatures(DPGSDataset dataset) {
		int numExs = dataset.getNumberOfExamples();
		for (int idxEx = 0; idxEx < numExs; ++idxEx) {
			// Current input structure.
			DPGSInput input = dataset.getInput(idxEx);

			// Number of tokens within the current input.
			int numTkns = input.size();

			for (int idxHead = 0; idxHead < numTkns; ++idxHead) {
				// Grandparent features.
				for (int idxModifier = 0; idxModifier < numTkns; ++idxModifier) {
					for (int idxGrandparent = 0; idxGrandparent < numTkns; ++idxGrandparent) {
						// Skip non-existent edges.
						if (input.getBasicGrandparentFeatures(idxHead,
								idxModifier, idxGrandparent) == null)
							continue;

						// List of generated features for the current factor.
						LinkedList<Integer> ftrs = new LinkedList<Integer>();

						/*
						 * Instantiate edge features and add them to active
						 * features list.
						 */
						for (int idxTpl = 0; idxTpl < grandparentTemplates.length; ++idxTpl) {
							DPGrandparentTemplate tpl = (DPGrandparentTemplate) grandparentTemplates[idxTpl];
							try {
								tpl.instantiateGrandparentDerivedFeatures(
										input, ftrs, explicitEncoding, idxHead,
										idxModifier, idxGrandparent);
							} catch (CloneNotSupportedException e) {
								LOG.error("Instantiating feature", e);
							}
						}

						// Set feature vector of this input.
						int numFtrs = ftrs.size();
						int[] ftrVals = new int[numFtrs];
						Iterator<Integer> itFtrVals = ftrs.iterator();
						for (int idxFtr = 0; idxFtr < numFtrs; ++idxFtr)
							ftrVals[idxFtr] = itFtrVals.next();
						input.setGrandparentFeatures(idxHead, idxModifier,
								idxGrandparent, ftrVals);
					}
				}

				// Left siblings features.
				for (int idxModifier = 0; idxModifier <= idxHead; ++idxModifier) {
					instantiateSiblingsFeatures(input, idxHead, idxModifier,
							idxHead);
					for (int idxPrevModifier = 0; idxPrevModifier < idxModifier; ++idxPrevModifier)
						instantiateSiblingsFeatures(input, idxHead,
								idxModifier, idxPrevModifier);
				}

				// Right siblings features.
				for (int idxModifier = idxHead + 1; idxModifier <= numTkns; ++idxModifier) {
					instantiateSiblingsFeatures(input, idxHead, idxModifier,
							numTkns);
					for (int idxPrevModifier = 0; idxPrevModifier < idxModifier; ++idxPrevModifier)
						instantiateSiblingsFeatures(input, idxHead,
								idxModifier, idxPrevModifier);
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

	@Override
	public void save(String fileName, Dataset dataset) throws IOException,
			FileNotFoundException {
		throw new NotImplementedException();
	}

	/**
	 * Return the number of instantiated parameters. Roughly, that is the number
	 * of parameters with value different from zero.
	 * 
	 * @return
	 */
	public int getNumberOfUpdatedParameters() {
		return parameters.size();
	}
}
