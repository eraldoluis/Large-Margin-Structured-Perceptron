package br.pucrio.inf.learn.structlearning.discriminative.evaluation;

import java.util.Collection;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;
import java.util.TreeMap;

import br.pucrio.inf.learn.structlearning.discriminative.data.ExampleInput;
import br.pucrio.inf.learn.structlearning.discriminative.data.ExampleOutput;

/**
 * Abstract class that provides methods to evaluate precision, recall and F1
 * values of examples that represent entities or examples where one can extract
 * entities from. Users of this class must implement the decoding (
 * <code>extractEntities</code>) and encoding (<code>tagEntity</code>).
 * Additionally, she should also implement the method <code>clearOutput</code>
 * that erases any encoded entity in a given output structure.
 * 
 * @author eraldof
 * 
 */
public abstract class EntityF1Evaluation {

	/**
	 * Valid chunk types (labels without the B- or I- part). One can use this
	 * property to ignore some chunk types.
	 */
	private Set<String> validChunkTypes;

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
	 * Extract entities from the correct pairs of input/output structures (
	 * <code>inputs</code> and <code>corrects</codes>) 
	 * and the predicted ones (<code>inputs</code> and <code>predicteds</code>).
	 * Then, calculate the F1-measure (and precision and recall) for each type
	 * of entity and the overall performance using micro averaging.
	 * 
	 * The returned map contains one performance measure for each type of entity
	 * whose keys are the type itself and the overall performance with the key
	 * 'overall'.
	 * 
	 * @param inputs
	 * @param corrects
	 * @param predicteds
	 * @return
	 */
	public Map<String, F1Measure> evaluateExamples(ExampleInput[] inputs,
			ExampleOutput[] corrects, ExampleOutput[] predicteds) {

		// Temporary sets to store the correct and predicted entities within
		// each example.
		HashSet<TypedEntity> correctEntities = new HashSet<TypedEntity>();
		HashSet<TypedEntity> predictedEntities = new HashSet<TypedEntity>();

		// Store the results for each class and one more for the overall
		// performance.
		TreeMap<String, F1Measure> res = new TreeMap<String, F1Measure>();

		// The overall performance.
		F1Measure overall = new F1Measure("overall");
		res.put(overall.getCaption(), overall);

		// Evaluate each sentence.
		for (int idxSeq = 0; idxSeq < corrects.length; ++idxSeq) {
			ExampleInput inputSeq = inputs[idxSeq];
			ExampleOutput correctSeq = corrects[idxSeq];
			ExampleOutput predictedSeq = predicteds[idxSeq];

			// Extract the correct entities.
			decodeEntities(inputSeq, correctSeq, correctEntities);

			// Extract the predicted entities.
			decodeEntities(inputSeq, predictedSeq, predictedEntities);

			// Count the total number of entities (nobjects) and the number of
			// correctly identified entities (nfullycorrect).
			for (TypedEntity ent : correctEntities) {
				// Skip ignored chunk types.
				if (validChunkTypes != null
						&& !validChunkTypes.contains(ent.getType()))
					continue;

				// Get (or create) the result for this type.
				F1Measure resCurClass = getResultByClass(res, ent.getType());

				resCurClass.incNumObjects();
				overall.incNumObjects();

				if (predictedEntities.contains(ent)) {
					resCurClass.incNumCorrectlyPredicted();
					overall.incNumCorrectlyPredicted();
				}
			}

			// Count the number of misidentified entities.
			for (TypedEntity ent : predictedEntities) {
				// Skip ignored chunk types.
				if (validChunkTypes != null
						&& !validChunkTypes.contains(ent.getType()))
					continue;

				F1Measure resCurClass = getResultByClass(res, ent.getType());
				resCurClass.incNumPredicted();
				overall.incNumPredicted();
			}

			// Clear the sets.
			correctEntities.clear();
			predictedEntities.clear();
		}

		return res;
	}

	/**
	 * Return the result within the given map for the given class. If the map
	 * does not include such a result, create a new one and include it in the
	 * map.
	 * 
	 * @param map
	 * @param type
	 * @return
	 */
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
	 * @param input
	 *            the input sequence. Usually, this value is not necessary to
	 *            extract the chunks.
	 * @param output
	 *            the output sequence.
	 * @param entities
	 *            the extracted chunks will be added to this collection.
	 */
	public abstract void decodeEntities(ExampleInput input,
			ExampleOutput output, Collection<TypedEntity> entities);

	/**
	 * Encode the given list of entities in the given output structure.
	 * Optionally, clear the output structure before encoding the entities.
	 * 
	 * @param input
	 * @param output
	 * @param entities
	 * @param clear
	 */
	public void encodeEntities(ExampleInput input, ExampleOutput output,
			Iterable<TypedEntity> entities, boolean clear) {
		// Optionally clear the output structure before encoding the entities.
		if (clear)
			clearEncodedEntities(input, output);

		// Encode the entities.
		for (TypedEntity chunk : entities)
			encodeEntity(input, output, chunk);
	}

	/**
	 * Encode the given entity in the given output structure.
	 * 
	 * @param input
	 * @param output
	 * @param entity
	 */
	public abstract void encodeEntity(ExampleInput input, ExampleOutput output,
			TypedEntity entity);

	/**
	 * Clear any encoded entity in the given output structure.
	 * 
	 * @param input
	 * @param output
	 */
	public abstract void clearEncodedEntities(ExampleInput input,
			ExampleOutput output);
}
