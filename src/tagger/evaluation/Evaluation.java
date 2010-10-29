package tagger.evaluation;

import java.util.Collection;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedList;
import java.util.Map;
import java.util.Set;

import tagger.data.Dataset;
import tagger.data.DatasetExample;
import tagger.data.DatasetException;
import tagger.learning.Verbose_res;

public class Evaluation {
	private String nullTag;
	private Set<String> validChunkTypes;

	public Evaluation(String nullTag) {
		this.nullTag = nullTag;
		this.validChunkTypes = null;
	}

	public void setValidChunkTypes(Iterable<String> listOfValidChunkTypes) {
		validChunkTypes = new HashSet<String>();
		for (String validType : listOfValidChunkTypes)
			validChunkTypes.add(validType);
	}

	public void setValidChunkTypes(String[] listOfValidChunkTypes) {
		validChunkTypes = new HashSet<String>();
		for (String validType : listOfValidChunkTypes)
			validChunkTypes.add(validType);
	}

	/**
	 * Evaluate the predicted tags on the dataset.
	 * 
	 * @param dataset
	 *            a dataset containing the golden sequence of tags.
	 * @param predictedTags
	 *            the predicted sequence of tags.
	 * @param tagset
	 *            the tagset used within the dataset and also within the
	 *            predicted sequence of tags.
	 * 
	 * @return a map of the measured performances, one for each class plus one
	 *         for the overall performance.
	 * 
	 * @throws DatasetException
	 */
	public Map<String, Verbose_res> evaluateSequences(Dataset dataset,
			String goldFeatureLabel, String predictedFeatureLabel)
			throws DatasetException {

		int goldFeature = dataset.getFeatureIndex(goldFeatureLabel);
		int predictedFeature = dataset.getFeatureIndex(predictedFeatureLabel);

		HashSet<TypedChunk> correct = new HashSet<TypedChunk>();
		HashSet<TypedChunk> predicted = new HashSet<TypedChunk>();

		// Store the results for each class and one more for the overall
		// performance.
		HashMap<String, Verbose_res> res = new HashMap<String, Verbose_res>();

		Verbose_res overall = new Verbose_res("overall");
		res.put(overall.L, overall);

		int numSentences = dataset.getNumberOfExamples();

		// Evaluate each sentence.
		for (int idxSent = 0; idxSent < numSentences; ++idxSent) {
			// Get the sentence.
			DatasetExample example = dataset.getExample(idxSent);

			// Extract the correct entities.
			extractEntities(example, goldFeature, correct);

			// Extract the predicted entities.
			extractEntities(example, predictedFeature, predicted);

			// Count the total number of entities (nobjects) and the number of
			// correctly identified entities (nfullycorrect).
			for (TypedChunk ent : correct) {
				Verbose_res resCurClass = getResultByClass(res, ent.type);
				++resCurClass.nobjects;
				++overall.nobjects;

				if (predicted.contains(ent)) {
					++resCurClass.nfullycorrect;
					++overall.nfullycorrect;
				}
			}

			// Count the number of misidentified entities.
			for (TypedChunk ent : predicted) {
				Verbose_res resCurClass = getResultByClass(res, ent.type);
				++resCurClass.nanswers;
				++overall.nanswers;
			}

			// Clear the sets.
			correct.clear();
			predicted.clear();
		}

		return res;
	}

	private Verbose_res getResultByClass(HashMap<String, Verbose_res> map,
			String type) {
		Verbose_res res = map.get(type);
		if (res == null) {
			res = new Verbose_res(type);
			map.put(type, res);
		}
		return res;
	}

	/**
	 * Extract the entities codified in the given tagged example.
	 * 
	 * The entities must be encoded using the IOB2 tagging style. The entities
	 * are encoded in the given feature.
	 * 
	 * @param dataset
	 *            tagged dataset.
	 * @param featureLabel
	 *            label of the feature from where to extract the entities.
	 * 
	 * @return a collection with the extracted entities.
	 * 
	 * @throws DatasetException
	 *             if the given feature label does not exist.
	 */
	public Collection<TypedChunk> extractEntities(Dataset dataset,
			String featureLabel) throws DatasetException {
		int idxFeature = dataset.getFeatureIndex(featureLabel);
		LinkedList<TypedChunk> list = new LinkedList<TypedChunk>();
		for (DatasetExample example : dataset)
			extractEntities(example, idxFeature, list);
		return list;
	}

	/**
	 * Extract the entities tagged in the given example.
	 * 
	 * @param example
	 * @param featureLabel
	 * @return
	 * @throws DatasetException
	 */
	public Collection<TypedChunk> extractEntities(DatasetExample example,
			String featureLabel) throws DatasetException {
		return extractEntities(example,
				example.getDataset().getFeatureIndex(featureLabel));
	}

	/**
	 * Extract the entities codified in the given tagged example.
	 * 
	 * The entities must be encoded using the IOB2 tagging style. The entities
	 * are encoded in the given feature.
	 * 
	 * @param example
	 *            a tagged example.
	 * @param feature
	 *            the index of the feature where the entities are encoded.
	 * 
	 * @return a <code>Collection</code> with all extracted entities.
	 */
	public Collection<TypedChunk> extractEntities(DatasetExample example,
			int feature) {
		LinkedList<TypedChunk> list = new LinkedList<TypedChunk>();
		extractEntities(example, feature, list);
		return list;
	}

	/**
	 * Extract the entities codified in the given tag sequence.
	 * 
	 * The entities must be codified using the IOB2 tagging style.
	 * 
	 * @param idxSentence
	 *            the index of the given sentence.
	 * @param tags
	 *            the tag sequence from where extract the entities.
	 * @param entities
	 *            the collection where the entities will be stored.
	 * @param tagset
	 *            the tagset used in the given sequence of tags.
	 */
	public void extractEntities(DatasetExample example, int feature,
			Collection<TypedChunk> entities) {

		int idxTknBegin = 0;
		String curType = nullTag;

		int lenExample = example.size();
		for (int idxTkn = 0; idxTkn < lenExample; ++idxTkn) {
			String tag = example.getFeatureValueAsString(idxTkn, feature);

			String beg = nullTag;
			String type = nullTag;

			String[] strs = tag.split("-", 2);
			beg = strs[0];
			if (strs.length > 1)
				type = strs[1];
			else
				type = beg;

			// Find the begining of an entity (maybe an "O entity").
			if (!type.equals(curType) || beg.equals("B")) {
				// If the previous entity is a valid one (not "O entity").
				if (!curType.equals(nullTag)) {
					if (validChunkTypes == null
							|| validChunkTypes.contains(curType))
						entities.add(new TypedChunk(example.getIndex(),
								idxTknBegin, idxTkn - 1, curType));
				}

				// Restart the current entity.
				idxTknBegin = idxTkn;
				curType = type;
			}
		}

		// If the last entity ends at the last token of the sentence.
		if (!curType.equals(nullTag)) {
			if (validChunkTypes == null || validChunkTypes.contains(curType))
				entities.add(new TypedChunk(example.getIndex(), idxTknBegin,
						lenExample - 1, curType));
		}
	}

	/**
	 * Extract the entities from the given example and store in the given map
	 * that is organized by entity type.
	 * 
	 * @param example
	 * @param featureLabel
	 * @param entitiesByType
	 * @return
	 * @throws DatasetException
	 */
	public int extractEntitiesByType(DatasetExample example,
			String featureLabel,
			Map<String, LinkedList<TypedChunk>> entitiesByType)
			throws DatasetException {
		return extractEntitiesByType(example, example.getDataset()
				.getFeatureIndex(featureLabel), entitiesByType);
	}

	/**
	 * Extract the entities from the given example and store in the given map
	 * that is organized by entity type.
	 * 
	 * @param example
	 * @param feature
	 * @param entitiesByType
	 * @return the number of entities extracted.
	 */
	public int extractEntitiesByType(DatasetExample example, int feature,
			Map<String, LinkedList<TypedChunk>> entitiesByType) {
		int numEntities = 0;
		int idxTknBegin = 0;
		String curType = nullTag;

		int lenExample = example.size();
		for (int idxTkn = 0; idxTkn < lenExample; ++idxTkn) {
			String tag = example.getFeatureValueAsString(idxTkn, feature);

			String beg = nullTag;
			String type = nullTag;

			String[] strs = tag.split("-", 2);
			beg = strs[0];
			if (strs.length > 1)
				type = strs[1];
			else
				type = beg;

			// Find the begining of an entity (maybe an "O entity").
			if (!type.equals(curType) || beg.equals("B")) {
				// If the previous entity is a valid one (not "O entity").
				if (!curType.equals(nullTag)) {
					if (validChunkTypes == null
							|| validChunkTypes.contains(curType)) {
						addEntityByType(new TypedChunk(example.getIndex(),
								idxTknBegin, idxTkn - 1, curType),
								entitiesByType);
						++numEntities;
					}
				}

				// Restart the current entity.
				idxTknBegin = idxTkn;
				curType = type;
			}
		}

		// If the last entity ends at the last token of the sentence.
		if (!curType.equals(nullTag)) {
			if (validChunkTypes == null || validChunkTypes.contains(curType)) {
				addEntityByType(new TypedChunk(example.getIndex(), idxTknBegin,
						lenExample - 1, curType), entitiesByType);
				++numEntities;
			}
		}

		return numEntities;
	}

	/**
	 * Add the given typed chunk in one of the lists in the given map, according
	 * to the chunk type. If there is no list for the corresponding chunk type,
	 * then create it and add it to the map.
	 * 
	 * @param typedChunk
	 * @param entitiesByType
	 */
	private void addEntityByType(TypedChunk typedChunk,
			Map<String, LinkedList<TypedChunk>> entitiesByType) {
		LinkedList<TypedChunk> entities = entitiesByType.get(typedChunk.type);
		if (entities == null) {
			entities = new LinkedList<TypedChunk>();
			entitiesByType.put(typedChunk.type, entities);
		}

		entities.add(typedChunk);
	}

	/**
	 * Encode <code>entities</code> in the feature labeled
	 * <code>featureLabel</code> of the given dataset.
	 * 
	 * @param dataset
	 * @param featureLabel
	 * @param entities
	 * @param cleanFeature
	 * @param alwaysUseBTag
	 * @throws DatasetException
	 */
	public void tagEntities(Dataset dataset, String featureLabel,
			Iterable<TypedChunk> entities, boolean cleanFeature,
			boolean alwaysUseBTag) throws DatasetException {
		// Encode the null tag.
		int nullTagCode = dataset.getFeatureValueEncoding().putString(nullTag);
		int feature = dataset.getFeatureIndex(featureLabel);

		// Clean the feature before tagging the entities.
		if (cleanFeature)
			for (DatasetExample example : dataset)
				for (int tkn = 0; tkn < example.size(); ++tkn)
					example.setFeatureValue(tkn, feature, nullTagCode);

		// Tag the entities.
		for (TypedChunk entity : entities)
			tagEntity(dataset.getExample(entity.sentence), feature, entity,
					nullTagCode, alwaysUseBTag);
	}

	/**
	 * Encode <code>entities</code> in the feature labeled
	 * <code>featureLabel</code> of the given dataset.
	 * 
	 * @param example
	 * @param featureLabel
	 * @param entities
	 * @param cleanFeature
	 * @param alwaysUseBTag
	 * @throws DatasetException
	 */
	public void tagEntities(DatasetExample example, String featureLabel,
			Iterable<TypedChunk> entities, boolean cleanFeature,
			boolean alwaysUseBTag) throws DatasetException {
		int feature = example.getDataset().getFeatureIndex(featureLabel);

		// Null-tag code;
		int nullTagCode = example.getFeatureEncoding().putString(nullTag);

		// Clean the feature before tagging the entities.
		if (cleanFeature)
			for (int tkn = 0; tkn < example.size(); ++tkn)
				example.setFeatureValue(tkn, feature, nullTagCode);

		// Tag the entities.
		for (TypedChunk entity : entities)
			tagEntity(example, feature, entity, nullTagCode, alwaysUseBTag);
	}

	public void tagEntities(DatasetExample example, int feature,
			Iterable<TypedChunk> entities, boolean cleanFeature,
			boolean alwaysUseBTag) throws DatasetException {
		// Null-tag code;
		int nullTagCode = example.getFeatureEncoding().putString(nullTag);

		// Clean the feature before tagging the entities.
		if (cleanFeature)
			for (int tkn = 0; tkn < example.size(); ++tkn)
				example.setFeatureValue(tkn, feature, nullTagCode);

		// Tag the entities.
		for (TypedChunk entity : entities)
			tagEntity(example, feature, entity, nullTagCode, alwaysUseBTag);
	}

	private void tagEntity(DatasetExample example, int feature,
			TypedChunk entity, int nullTagCode, boolean alwaysUseBTag)
			throws DatasetException {
		// Prefix strings.
		String beg, in;
		if (entity.type.length() == 0) {
			beg = "B";
			in = "I";
		} else {
			beg = "B-";
			in = "I-";
		}

		String firstPrefix = beg;
		if (!alwaysUseBTag) {
			if (entity.tokenBeg == 0)
				firstPrefix = in;
			else {
				String prevTag = example.getFeatureValueAsString(
						entity.tokenBeg - 1, feature);
				if (!prevTag.equals(beg + entity.type)
						&& !prevTag.equals(in + entity.type))
					firstPrefix = in;
			}
		}

		example.setFeatureValue(entity.tokenBeg, feature, firstPrefix
				+ entity.type);
		for (int tkn = entity.tokenBeg + 1; tkn <= entity.tokenEnd; ++tkn)
			example.setFeatureValue(tkn, feature, in + entity.type);
	}
}
