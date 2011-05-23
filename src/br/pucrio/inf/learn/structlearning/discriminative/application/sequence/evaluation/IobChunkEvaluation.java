package br.pucrio.inf.learn.structlearning.discriminative.application.sequence.evaluation;

import java.util.Collection;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;
import java.util.TreeMap;

import br.pucrio.inf.learn.structlearning.discriminative.application.sequence.SequenceInput;
import br.pucrio.inf.learn.structlearning.discriminative.application.sequence.SequenceOutput;
import br.pucrio.inf.learn.structlearning.discriminative.application.sequence.data.DatasetException;
import br.pucrio.inf.learn.structlearning.discriminative.data.FeatureEncoding;

/**
 * Provide methods to evaluate precision, recall and F1 values of sequences that
 * contain chunks codified with IOB tagging style.
 * 
 * @author eraldof
 * 
 */
public class IobChunkEvaluation {

	/**
	 * Label that codifies no chunk information.
	 */
	private String nullLabel;

	/**
	 * Valid chunk types (labels without the B- or I- part). One can use this
	 * property to ignore some chunk types.
	 */
	private Set<String> validChunkTypes;

	/**
	 * The encoding for state labels.
	 */
	private FeatureEncoding<String> stateEncoding;

	/**
	 * Create an evaluation object. The user must provide the state-label
	 * encoding and the null label.
	 * 
	 * @param stateEncoding
	 * @param nullLabel
	 */
	public IobChunkEvaluation(FeatureEncoding<String> stateEncoding,
			String nullLabel) {
		this.stateEncoding = stateEncoding;
		this.nullLabel = nullLabel;
	}

	/**
	 * Set the list of valid chunk types. If this list is non-null then only
	 * chunks of these types are considered in the evaluation. Otherwise, all
	 * chunks found in the output sequences are considered.
	 * 
	 * @param listOfValidChunkTypes
	 */
	public void setValidChunkTypes(Iterable<String> listOfValidChunkTypes) {
		if (listOfValidChunkTypes == null) {
			validChunkTypes = null;
			return;
		}

		validChunkTypes = new HashSet<String>();
		for (String validType : listOfValidChunkTypes)
			validChunkTypes.add(validType);
	}

	/**
	 * Set the list of valid chunk types. If this list is non-null then only
	 * chunks of these types are considered in the evaluation. Otherwise, all
	 * chunks found in the output sequences are considered.
	 * 
	 * @param listOfValidChunkTypes
	 */
	public void setValidChunkTypes(String[] listOfValidChunkTypes) {
		if (listOfValidChunkTypes == null) {
			validChunkTypes = null;
			return;
		}

		validChunkTypes = new HashSet<String>();
		for (String validType : listOfValidChunkTypes)
			validChunkTypes.add(validType);
	}

	/**
	 * Return the rate between the number of correctly classified tokens and the
	 * total number of tokens.
	 * 
	 * @param inputSeqs
	 * @param correctSeqs
	 * @param predictedSeqs
	 * @return
	 */
	public double evaluateAccuracy(SequenceInput[] inputSeqs,
			SequenceOutput[] correctSeqs, SequenceOutput[] predictedSeqs) {
		int total = 0;
		int correct = 0;
		for (int idxSeq = 0; idxSeq < correctSeqs.length; ++idxSeq) {
			for (int idxTkn = 0; idxTkn < correctSeqs[idxSeq].size(); ++idxTkn) {
				++total;
				if (correctSeqs[idxSeq].getLabel(idxTkn) == predictedSeqs[idxSeq]
						.getLabel(idxTkn))
					++correct;
			}
		}
		return ((double) correct) / total;
	}

	/**
	 * Evaluate the precision and recall between the chunks codified in a
	 * correct output sequence and a predicted sequence.
	 * 
	 * @param inputSeqs
	 * @param correctSeqs
	 * @param predictedSeqs
	 * @return
	 */
	public Map<String, F1Measure> evaluateSequences(SequenceInput[] inputSeqs,
			SequenceOutput[] correctSeqs, SequenceOutput[] predictedSeqs) {

		HashSet<TypedChunk> correctCks = new HashSet<TypedChunk>();
		HashSet<TypedChunk> predictedCks = new HashSet<TypedChunk>();

		// Store the results for each class and one more for the overall
		// performance.
		TreeMap<String, F1Measure> res = new TreeMap<String, F1Measure>();

		F1Measure overall = new F1Measure("overall");
		res.put(overall.getCaption(), overall);

		// Evaluate each sentence.
		for (int idxSeq = 0; idxSeq < correctSeqs.length; ++idxSeq) {
			SequenceInput inputSeq = inputSeqs[idxSeq];
			SequenceOutput correctSeq = correctSeqs[idxSeq];
			SequenceOutput predictedSeq = predictedSeqs[idxSeq];

			// Extract the correct entities.
			extractEntities(idxSeq, inputSeq, correctSeq, correctCks);

			// Extract the predicted entities.
			extractEntities(idxSeq, inputSeq, predictedSeq, predictedCks);

			// Count the total number of entities (nobjects) and the number of
			// correctly identified entities (nfullycorrect).
			for (TypedChunk ent : correctCks) {
				F1Measure resCurClass = getResultByClass(res, ent.type);
				resCurClass.incNumObjects();
				overall.incNumObjects();

				if (predictedCks.contains(ent)) {
					resCurClass.incNumCorrectlyPredicted();
					overall.incNumCorrectlyPredicted();
				}
			}

			// Count the number of misidentified entities.
			for (TypedChunk ent : predictedCks) {
				F1Measure resCurClass = getResultByClass(res, ent.type);
				resCurClass.incNumPredicted();
				overall.incNumPredicted();
			}

			// Clear the sets.
			correctCks.clear();
			predictedCks.clear();
		}

		return res;
	}

	private F1Measure getResultByClass(Map<String, F1Measure> map, String type) {
		F1Measure res = map.get(type);
		if (res == null) {
			res = new F1Measure(type);
			map.put(type, res);
		}
		return res;
	}

	/**
	 * Extract the chunks within the given output sequence.
	 * 
	 * @param sequenceIndex
	 *            the index of the given sequence. It is assigned to every
	 *            extracted chunk.
	 * @param input
	 *            the input sequence. Usually, this value is not necessary to
	 *            extract the chunks.
	 * @param output
	 *            the output sequence.
	 * @param chunks
	 *            the extracted chunks will be added to this collection.
	 */
	public void extractEntities(int sequenceIndex, SequenceInput input,
			SequenceOutput output, Collection<TypedChunk> chunks) {

		int idxTknBegin = 0;
		String curType = nullLabel;

		int lenExample = output.size();
		for (int idxTkn = 0; idxTkn < lenExample; ++idxTkn) {
			String tag = stateEncoding.getValueByCode(output.getLabel(idxTkn));

			String beg = nullLabel;
			String type = nullLabel;

			String[] strs = tag.split("-", 2);
			beg = strs[0];
			if (strs.length > 1)
				type = strs[1];
			else
				type = beg;

			// Find the begining of an entity (maybe an "O entity").
			if (!type.equals(curType) || beg.equals("B")) {
				// If the previous entity is a valid one (not "O entity").
				if (!curType.equals(nullLabel)) {
					if (validChunkTypes == null
							|| validChunkTypes.contains(curType))
						chunks.add(new TypedChunk(sequenceIndex, idxTknBegin,
								idxTkn - 1, curType));
				}

				// Restart the current entity.
				idxTknBegin = idxTkn;
				curType = type;
			}
		}

		// If the last entity ends at the last token of the sentence.
		if (!curType.equals(nullLabel)) {
			if (validChunkTypes == null || validChunkTypes.contains(curType))
				chunks.add(new TypedChunk(sequenceIndex, idxTknBegin,
						lenExample - 1, curType));
		}
	}

	// public int extractEntitiesByType(DatasetExample example, int feature,
	// Map<String, LinkedList<TypedChunk>> entitiesByType) {
	// int numEntities = 0;
	// int idxTknBegin = 0;
	// String curType = nullTag;
	//
	// int lenExample = example.size();
	// for (int idxTkn = 0; idxTkn < lenExample; ++idxTkn) {
	// String tag = example.getFeatureValueAsString(idxTkn, feature);
	//
	// String beg = nullTag;
	// String type = nullTag;
	//
	// String[] strs = tag.split("-", 2);
	// beg = strs[0];
	// if (strs.length > 1)
	// type = strs[1];
	// else
	// type = beg;
	//
	// // Find the begining of an entity (maybe an "O entity").
	// if (!type.equals(curType) || beg.equals("B")) {
	// // If the previous entity is a valid one (not "O entity").
	// if (!curType.equals(nullTag)) {
	// if (validChunkTypes == null
	// || validChunkTypes.contains(curType)) {
	// addEntityByType(new TypedChunk(example.getIndex(),
	// idxTknBegin, idxTkn - 1, curType),
	// entitiesByType);
	// ++numEntities;
	// }
	// }
	//
	// // Restart the current entity.
	// idxTknBegin = idxTkn;
	// curType = type;
	// }
	// }
	//
	// // If the last entity ends at the last token of the sentence.
	// if (!curType.equals(nullTag)) {
	// if (validChunkTypes == null || validChunkTypes.contains(curType)) {
	// addEntityByType(new TypedChunk(example.getIndex(), idxTknBegin,
	// lenExample - 1, curType), entitiesByType);
	// ++numEntities;
	// }
	// }
	//
	// return numEntities;
	// }

	// private void addEntityByType(TypedChunk typedChunk,
	// Map<String, LinkedList<TypedChunk>> entitiesByType) {
	// LinkedList<TypedChunk> entities = entitiesByType.get(typedChunk.type);
	// if (entities == null) {
	// entities = new LinkedList<TypedChunk>();
	// entitiesByType.put(typedChunk.type, entities);
	// }
	//
	// entities.add(typedChunk);
	// }

	// public void tagEntities(Dataset dataset, String featureLabel,
	// Iterable<TypedChunk> entities, boolean cleanFeature,
	// boolean alwaysUseBTag) throws DatasetException {
	// // Encode the null tag.
	// int nullTagCode = dataset.getFeatureValueEncoding().putString(nullTag);
	// int feature = dataset.getFeatureIndex(featureLabel);
	//
	// // Clean the feature before tagging the entities.
	// if (cleanFeature)
	// for (DatasetExample example : dataset)
	// for (int tkn = 0; tkn < example.size(); ++tkn)
	// example.setFeatureValue(tkn, feature, nullTagCode);
	//
	// // Tag the entities.
	// for (TypedChunk entity : entities)
	// tagEntity(dataset.getExample(entity.sentence), feature, entity,
	// nullTagCode, alwaysUseBTag);
	// }

	public void tagEntities(SequenceInput inputSeq, SequenceOutput outputSeq,
			Iterable<TypedChunk> chunks, boolean cleanFeature,
			boolean alwaysUseBTag) throws DatasetException {
		// Null-tag code;
		int nullTagCode = stateEncoding.put(nullLabel);

		// Clean the feature before tagging the entities.
		if (cleanFeature)
			for (int tkn = 0; tkn < outputSeq.size(); ++tkn)
				outputSeq.setLabel(tkn, nullTagCode);

		// Tag the entities.
		for (TypedChunk chunk : chunks)
			tagEntity(inputSeq, outputSeq, chunk, nullTagCode, alwaysUseBTag);
	}

	/**
	 * Tag the output sequence with the given chunk.
	 * 
	 * @param inputSeq
	 * @param outputSeq
	 * @param chunk
	 * @param nullTagCode
	 * @param alwaysUseBTag
	 */
	private void tagEntity(SequenceInput inputSeq, SequenceOutput outputSeq,
			TypedChunk chunk, int nullTagCode, boolean alwaysUseBTag) {
		// Label prefixes.
		String beg, in;
		if (chunk.type.length() == 0) {
			beg = "B";
			in = "I";
		} else {
			beg = "B-";
			in = "I-";
		}

		// Check the first token tag (B- or I-).
		String firstPrefix = beg;
		if (!alwaysUseBTag) {
			if (chunk.tokenBeg == 0)
				firstPrefix = in;
			else {
				String prevTag = stateEncoding.getValueByCode(outputSeq
						.getLabel(chunk.tokenBeg - 1));
				if (!prevTag.equals(beg + chunk.type)
						&& !prevTag.equals(in + chunk.type))
					firstPrefix = in;
			}
		}

		// Tag the output sequence with the given entity.
		outputSeq.setLabel(chunk.tokenBeg,
				stateEncoding.put(firstPrefix + chunk.type));
		for (int tkn = chunk.tokenBeg + 1; tkn <= chunk.tokenEnd; ++tkn)
			outputSeq.setLabel(tkn, stateEncoding.put(in + chunk.type));
	}
}
