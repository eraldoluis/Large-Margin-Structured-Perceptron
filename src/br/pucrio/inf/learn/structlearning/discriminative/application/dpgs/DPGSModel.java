package br.pucrio.inf.learn.structlearning.discriminative.application.dpgs;

import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
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
import br.pucrio.inf.learn.structlearning.discriminative.data.ExampleInputArray;
import br.pucrio.inf.learn.structlearning.discriminative.data.ExampleOutput;
import br.pucrio.inf.learn.structlearning.discriminative.data.encoding.FeatureEncoding;
import br.pucrio.inf.learn.structlearning.discriminative.data.encoding.MapEncoding;
import br.pucrio.inf.learn.structlearning.discriminative.data.encoding.StringMapEncoding;
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
	 * Templates for edge-based features. Such features depend on two parameters
	 * only: a head token and a modifier token, that is a dependency edge.
	 */
	protected DPGSTemplate[] edgeTemplates;

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
	 * Grandparent templates that comprise three parameters: head token,
	 * modifier token and head of the head token (grandparent of modifier
	 * token).
	 */
	protected DPGSTemplate[] grandparentTemplates;

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

		// Deep-copy of parameters.
		for (Entry<Integer, AveragedParameter> entry : parameters.entrySet())
			entry.setValue(entry.getValue().clone());

		// Updated parameters and features are NOT copied.
		updatedParameters = new TreeSet<AveragedParameter>();

		// Explicit encoding just references the other one.
		this.explicitEncoding = other.explicitEncoding;

		// Templates.
		this.grandparentTemplates = other.grandparentTemplates;
		this.leftSiblingsTemplates = other.leftSiblingsTemplates;
		this.rightSiblingsTemplates = other.rightSiblingsTemplates;
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
	 * Update this model using the difference between the factors present in the
	 * correct (<code>outputCorrect</code>) and in the predicted output
	 * (</code>outputPredicted</code>). The correct factors are obtained from
	 * the parse tree scructure ( <code>outputCorrect.getHead(...)</code>).
	 * While the predicted factors are obtained from the grandparent structure (
	 * <code>outputPredicted.getGrandparent(...)</code>) and the modifier
	 * structure (<code>outputPredicted.isModifier(...)</code>).
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

			if (correctGrandparent != predictedGrandparent) {
				// Update edge factor parameter.
				loss += 1d; // Edge factor contribution.
				if (predictedGrandparent != -1)
					updateEdgeFactorParams(input, predictedGrandparent,
							idxHead, -learningRate);
				if (correctGrandparent != -1)
					updateEdgeFactorParams(input, correctGrandparent, idxHead,
							learningRate);
			}

			/*
			 * Verify grandparent and siblings factors for differences between
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
				 * Special tokens are always present, by definition.
				 */
				boolean isCorrectModifier = (isSpecialToken || (outputCorrect
						.getHead(idxModifier) == idxHead));
				boolean isPredictedModifier = (isSpecialToken || outputPredicted
						.isModifier(idxHead, idxModifier));

				if (!isCorrectModifier && !isPredictedModifier)
					/*
					 * Current modifier is neither included in the correct
					 * structure nor the predicted structure. Thus, skip it.
					 */
					continue;

				if (isCorrectModifier != isPredictedModifier) {
					//
					// Current modifier is misclassified.
					//

					// Siblings and grandparent factors contribution.
					loss += 2d;

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

				} else { // isCorrectModifier == isPredictedModifier
					/*
					 * The current modifier has been correctly predicted for the
					 * current head. Now, additionally check the previous
					 * modifier and the grandparent factor.
					 */

					if (correctPreviousModifier != predictedPreviousModifier) {
						// Siblings factor contribution.
						loss += 1d;

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

					if (!isSpecialToken
							&& correctGrandparent != predictedGrandparent) {
						// Grandparent factor contribution.
						loss += 1d;

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
					}
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
	 * Update all feature parameters in the given edge factor.
	 * 
	 * @param input
	 * @param idxHead
	 * @param idxModifier
	 * @param learnRate
	 */
	protected void updateEdgeFactorParams(DPGSInput input, int idxHead,
			int idxModifier, double learnRate) {
		int[] ftrs = input.getEdgeFeatures(idxHead, idxModifier);
		if (ftrs == null)
			// Inexistent factor. Do nothing.
			return;
		for (int idxFtr = 0; idxFtr < ftrs.length; ++idxFtr)
			updateFeatureParam(ftrs[idxFtr], learnRate);
	}

	/**
	 * Update all feature parameters in the given grandparent factor.
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
			// Inexistent factor. Do nothing.
			return;
		for (int idxFtr = 0; idxFtr < ftrs.length; ++idxFtr)
			updateFeatureParam(ftrs[idxFtr], learnRate);
	}

	/**
	 * Update all feature parameters in the given siblings factor.
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
		for (AveragedParameter param : parameters.values())
			param.average(numberOfIterations);
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

	public DPGSTemplate[] loadEdgeTemplates(BufferedReader reader,
			DPGSDataset dataset) throws IOException, DPGSException {
		LinkedList<DPGSTemplate> templatesList = new LinkedList<DPGSTemplate>();
		String line = DPGSDataset.skipBlanksAndComments(reader);
		while (line != null) {
			String[] ftrsStr = line.split("[ ]");
			int[] ftrs = new int[ftrsStr.length];
			for (int idx = 0; idx < ftrs.length; ++idx) {
				ftrs[idx] = dataset.getEdgeFeatureIndex(ftrsStr[idx]);
				if (ftrs[idx] == -1)
					throw new DPGSException(String.format(
							"Feature label %s does not exist", ftrsStr[idx]));
			}
			templatesList.add(new DPEdgeTemplate(templatesList.size(), ftrs));
			// Read next line.
			line = DPGSDataset.skipBlanksAndComments(reader);
		}

		// Convert list to array.
		return templatesList.toArray(new DPGSTemplate[0]);
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
	public DPGSTemplate[] loadGrandparentTemplates(BufferedReader reader,
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
		return templatesList.toArray(new DPGSTemplate[0]);
	}

	public void loadEdgeTemplates(String templatesFileName, DPGSDataset dataset)
			throws IOException, DPGSException {
		BufferedReader reader = new BufferedReader(new FileReader(
				templatesFileName));
		edgeTemplates = loadEdgeTemplates(reader, dataset);
		reader.close();
	}

	public void loadGrandparentTemplates(String templatesFileName,
			DPGSDataset dataset) throws IOException, DPGSException {
		BufferedReader reader = new BufferedReader(new FileReader(
				templatesFileName));
		grandparentTemplates = loadGrandparentTemplates(reader, dataset);
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
		for (int idxTpl = 0; idxTpl < leftSiblingsTemplates.length; ++idxTpl) {
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

	/**
	 * Generate derived features for the input structures in the given dataset (
	 * <code>dataset</code>). The derived features are specified by this model
	 * templates.
	 * 
	 * @param dataset
	 */
	public void generateFeatures(DPGSDataset dataset) {
		// // TODO test
		// int[] ftrs0 = new int[] { 0 };
		// int[] ftrs1 = new int[] { 1 };

		// Number of examples in the given dataset.
		int numExs = dataset.getNumberOfExamples();
		ExampleInputArray inputArray = dataset.getInputs();

		inputArray.loadInOrder();
		
		for (int idxEx = 0; idxEx < numExs; ++idxEx) {
			// Current input structure.
			DPGSInput input = (DPGSInput) inputArray.get(idxEx);
			generateFeaturesOneInput(input);

			// Progess report.
			if ((idxEx + 1) % 100 == 0) {
				System.out.print('.');
				System.out.flush();
			}
		}

		System.out.println();
		System.out.flush();
	}

	public void generateFeaturesOneInput(DPGSInput input) {
		// // TODO test
		// DPGSOutput output = dataset.getOutput(idxEx);

		// Number of tokens within the current input.
		int numTkns = input.size();

		for (int idxHead = 0; idxHead < numTkns; ++idxHead) {
			for (int idxModifier = 0; idxModifier < numTkns; ++idxModifier) {
				//
				// Edge features.
				//
				if (input.getBasicEdgeFeatures(idxHead, idxModifier) != null) {
					// List of generated features for the current factor.
					LinkedList<Integer> ftrs = new LinkedList<Integer>();

					/*
					 * Instantiate edge features and add them to active features
					 * list.
					 */
					for (int idxTpl = 0; idxTpl < edgeTemplates.length; ++idxTpl) {
						DPEdgeTemplate tpl = (DPEdgeTemplate) edgeTemplates[idxTpl];
						try {
							tpl.instantiateEdgeDerivedFeatures(input, ftrs,
									explicitEncoding, idxHead, idxModifier);
						} catch (CloneNotSupportedException e) {
							LOG.error("Instantiating feature", e);
						}
					}

					// Convert the list of feature codes to an array.
					int numFtrs = ftrs.size();
					int[] ftrVals = new int[numFtrs];
					Iterator<Integer> itFtrVals = ftrs.iterator();
					for (int idxFtr = 0; idxFtr < numFtrs; ++idxFtr)
						ftrVals[idxFtr] = itFtrVals.next();

					// Set feature vector of this input.
					input.setEdgeFeatures(idxHead, idxModifier, ftrVals);

					// // TODO test
					// if (output.getHead(idxModifier) == idxHead)
					// input.setEdgeFeatures(idxHead, idxModifier, ftrs1);
					// else
					// input.setEdgeFeatures(idxHead, idxModifier, ftrs0);
				}

				//
				// Grandparent features.
				//
				for (int idxGrandparent = 0; idxGrandparent < numTkns; ++idxGrandparent) {
					// Skip non-existent edges.
					if (input.getBasicGrandparentFeatures(idxHead, idxModifier,
							idxGrandparent) == null)
						continue;

					// List of generated features for the current factor.
					LinkedList<Integer> ftrs = new LinkedList<Integer>();

					/*
					 * Instantiate edge features and add them to active features
					 * list.
					 */
					for (int idxTpl = 0; idxTpl < grandparentTemplates.length; ++idxTpl) {
						DPGrandparentTemplate tpl = (DPGrandparentTemplate) grandparentTemplates[idxTpl];
						try {
							tpl.instantiateGrandparentDerivedFeatures(input,
									ftrs, explicitEncoding, idxHead,
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

					// // TODO test
					// if (output.getHead(idxModifier) == idxHead
					// && output.getHead(idxHead) == idxGrandparent)
					// input.setGrandparentFeatures(idxHead, idxModifier,
					// idxGrandparent, ftrs1);
					// else
					// input.setGrandparentFeatures(idxHead, idxModifier,
					// idxGrandparent, ftrs0);
				}
			}

			//
			// Left siblings features.
			//

			// // TODO test
			// int previousCorrectModifier = idxHead;

			for (int idxModifier = 0; idxModifier <= idxHead; ++idxModifier) {
				// Special factors: (idxHead, idxModifier, START).
				instantiateSiblingsFeatures(input, idxHead, idxModifier,
						idxHead);

				// // TODO test
				// boolean isModifier = (idxModifier == idxHead || output
				// .getHead(idxModifier) == idxHead);
				// if (isModifier && previousCorrectModifier == idxHead)
				// input.setSiblingsFeatures(idxHead, idxModifier,
				// idxHead, ftrs1);
				// else
				// input.setSiblingsFeatures(idxHead, idxModifier,
				// idxHead, ftrs0);

				// Remaining factors.
				for (int idxPrevModifier = 0; idxPrevModifier < idxModifier; ++idxPrevModifier) {
					instantiateSiblingsFeatures(input, idxHead, idxModifier,
							idxPrevModifier);

					// // TODO test
					// if (isModifier
					// && previousCorrectModifier == idxPrevModifier)
					// input.setSiblingsFeatures(idxHead, idxModifier,
					// idxPrevModifier, ftrs1);
					// else
					// input.setSiblingsFeatures(idxHead, idxModifier,
					// idxPrevModifier, ftrs0);
				}

				// // TODO test
				// if (isModifier)
				// previousCorrectModifier = idxModifier;
			}

			//
			// Right siblings features.
			//

			// // TODO teste
			// previousCorrectModifier = numTkns;

			for (int idxModifier = idxHead + 1; idxModifier <= numTkns; ++idxModifier) {
				// Special factors: (idxHead, idxModifier, START).
				instantiateSiblingsFeatures(input, idxHead, idxModifier,
						numTkns);

				// // TODO test
				// boolean isModifier = (idxModifier == numTkns || output
				// .getHead(idxModifier) == idxHead);
				// if (isModifier && previousCorrectModifier == numTkns)
				// input.setSiblingsFeatures(idxHead, idxModifier,
				// numTkns, ftrs1);
				// else
				// input.setSiblingsFeatures(idxHead, idxModifier,
				// numTkns, ftrs0);

				// Remaining factors.
				for (int idxPrevModifier = idxHead + 1; idxPrevModifier < idxModifier; ++idxPrevModifier) {
					instantiateSiblingsFeatures(input, idxHead, idxModifier,
							idxPrevModifier);

					// // TODO test
					// if (isModifier
					// && idxPrevModifier == previousCorrectModifier)
					// input.setSiblingsFeatures(idxHead, idxModifier,
					// idxPrevModifier, ftrs1);
					// else
					// input.setSiblingsFeatures(idxHead, idxModifier,
					// idxPrevModifier, ftrs0);
				}

				// // TODO test
				// if (isModifier)
				// previousCorrectModifier = idxModifier;
			}
		}
	}

	static public DPGSModelLoadReturn load(String fileName) throws ClassNotFoundException, IOException{
		FileInputStream input = null;
		BufferedInputStream bufInput = null;
		ObjectInputStream objInput = null;
		DPGSModel model;
		DPGSDataset dataset;
		
		try {
			input = new FileInputStream(fileName);
			bufInput = new BufferedInputStream(input);
			objInput = new ObjectInputStream(bufInput);
	
			int root = objInput.readInt();
			
			model = new DPGSModel(root);
			
			model.edgeTemplates = (DPGSTemplate[]) objInput.readObject();
			model.grandparentTemplates = (DPGSTemplate[]) objInput.readObject();
			model.leftSiblingsTemplates = (DPGSTemplate[]) objInput.readObject();
			model.rightSiblingsTemplates = (DPGSTemplate[]) objInput.readObject();
			model.parameters = (Map<Integer, AveragedParameter>) objInput.readObject();
			model.explicitEncoding = (MapEncoding<Feature>) objInput.readObject();
		
			dataset = DPGSDataset.loadCore(objInput);
		} finally {
			if(input != null)
				input.close();
		
			if(bufInput != null)
				bufInput.close();
			
			if(objInput != null)
				objInput.close();
		}
		
		return new DPGSModelLoadReturn(model, dataset);
	}
	
	@Override
	public void save(String fileName, Dataset dataset) throws IOException,
			FileNotFoundException {
		FileOutputStream output = null;
		BufferedOutputStream bufOutput = null;
		ObjectOutputStream objOut = null;
		
		try {
			output = new FileOutputStream(fileName);
			bufOutput = new BufferedOutputStream(output);
			objOut = new ObjectOutputStream(bufOutput);

			objOut.writeInt(this.root);
			objOut.writeObject(this.edgeTemplates);
			objOut.writeObject(this.grandparentTemplates);
			objOut.writeObject(this.leftSiblingsTemplates);
			objOut.writeObject(this.rightSiblingsTemplates);
			objOut.writeObject(this.parameters);
			objOut.writeObject(this.explicitEncoding);
			((DPGSDataset)dataset).saveCore(objOut);

			objOut.flush();
		} finally {
			
			if(objOut != null)
				objOut.close();
			else if(bufOutput != null)
				bufOutput.close();			
			else if(output != null)
					output.close();
			
		}
			
			
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
	
	private void loadExplicitEncoding() throws IOException, ClassNotFoundException {
		File file = new File("explicitEncoding");
		if (!file.exists()) {
			if (explicitEncoding == null)
				explicitEncoding = new MapEncoding<Feature>();
		}

		FileInputStream fileIn = null;
		BufferedInputStream buf = null;
		ObjectInputStream objInput = null;

		try {
			fileIn = new FileInputStream(file);
			buf = new BufferedInputStream(fileIn);
			objInput = new ObjectInputStream(buf);

			explicitEncoding = (MapEncoding<Feature>) objInput.readObject();
		} finally {
			if (objInput != null)
				objInput.close();
			else if (buf != null)
				buf.close();
			else if (fileIn != null)
				fileIn.close();
		}
	}

	private void unloadExplicitEncoding() throws IOException,
			ClassNotFoundException {
		
		FileOutputStream fileOut = null;
		BufferedOutputStream bufOut = null;
		ObjectOutputStream objOut = null;

		
		try{
			fileOut = new FileOutputStream("explicitEncoding");
			bufOut = new BufferedOutputStream(fileOut);
			objOut = new ObjectOutputStream(fileOut);

			objOut.writeObject(explicitEncoding);
		}finally{
			if (objOut != null)
				objOut.close();
			else if (bufOut != null)
				bufOut.close();
			else if (fileOut != null)
				fileOut.close();
		}
		
		explicitEncoding = null;
	}
	
	
	static public class DPGSModelLoadReturn{
		
		private DPGSModel model;
		private DPGSDataset dataset;
		
		private DPGSModelLoadReturn(DPGSModel model, DPGSDataset dataset) {
			super();
			this.model = model;
			this.dataset = dataset;
		}

		public DPGSModel getModel() {
			return model;
		}

		public DPGSDataset getDataset() {
			return dataset;
		}
	}
}
