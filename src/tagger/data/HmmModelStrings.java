package tagger.data;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.HashMap;
import java.util.Map.Entry;

/**
 * Represent an HMM using only the labels of states and observations, i.e., do
 * not encode neither states nor observations as integers.
 * 
 * @author eraldof
 * 
 */
public class HmmModelStrings {

	/**
	 * OUT state representation and also the default state.
	 */
	public static final String OUT_STATE = "0";

	public static final String FIRST_STATE_PREFIX = "$$FS$$";
	public static final String EMISSION_PREFIX = "$$E$$";
	public static final String TRANSITION_PREFIX = "$$T$$";

	private HashMap<String, Double> probFirstState;
	private HashMap<String, HashMap<String, Double>> probEmissions;
	private HashMap<String, HashMap<String, Double>> probTransitions;

	public HmmModelStrings() {
		probFirstState = new HashMap<String, Double>();
		probEmissions = new HashMap<String, HashMap<String, Double>>();
		probTransitions = new HashMap<String, HashMap<String, Double>>();
	}

	public HmmModelStrings(String fileName) throws IOException {
		probFirstState = new HashMap<String, Double>();
		probEmissions = new HashMap<String, HashMap<String, Double>>();
		probTransitions = new HashMap<String, HashMap<String, Double>>();

		loadFromHadoopModel(fileName);
	}

	public HmmModelStrings(BufferedReader reader) throws IOException {
		probFirstState = new HashMap<String, Double>();
		probEmissions = new HashMap<String, HashMap<String, Double>>();
		probTransitions = new HashMap<String, HashMap<String, Double>>();

		loadFromHadoopModel(reader);
	}

	public void loadFromHadoopModel(String fileName) throws IOException {
		FileReader reader = new FileReader(fileName);
		loadFromHadoopModel(new BufferedReader(reader));
		reader.close();
	}

	public void loadFromHadoopModel(BufferedReader reader) throws IOException {
		// Clear the data structures.
		probFirstState.clear();
		probEmissions.clear();
		probTransitions.clear();

		// State set.
		String buff;
		while ((buff = reader.readLine()) != null) {
			String[] keyAndValueSequence = buff.split("\\s");
			if (keyAndValueSequence.length < 1)
				continue;

			// Key of the distribution. Indicate which type (first state,
			// emission or transition) and which state is the distribution
			// related to.
			String keyOfDist = keyAndValueSequence[0];
			if (keyOfDist.equals(FIRST_STATE_PREFIX))
				fillDistribution(keyAndValueSequence, probFirstState);
			else if (keyOfDist.startsWith(EMISSION_PREFIX)) {
				String state = keyOfDist.substring(EMISSION_PREFIX.length());
				fillDistribution(keyAndValueSequence,
						getMapFromMap(probEmissions, state));
			} else if (keyOfDist.startsWith(TRANSITION_PREFIX)) {
				String state = keyOfDist.substring(TRANSITION_PREFIX.length());
				fillDistribution(keyAndValueSequence,
						getMapFromMap(probTransitions, state));
			}
		}
	}

	public void incEmissionCount(String state, String symbol) {
		HashMap<String, Double> map = getMapFromMap(probEmissions, state);
		incMapEntry(map, symbol, 1.0);
	}

	public void incTransitionCount(String stateFrom, String stateTo) {
		HashMap<String, Double> map = getMapFromMap(probTransitions, stateFrom);
		incMapEntry(map, stateTo, 1.0);
	}

	public void incFirstStateCount(String state) {
		incMapEntry(probFirstState, state, 1.0);
	}

	public void incEmissionCount(String state, String symbol, double val) {
		HashMap<String, Double> map = getMapFromMap(probEmissions, state);
		incMapEntry(map, symbol, val);
	}

	public void incTransitionCount(String stateFrom, String stateTo, double val) {
		HashMap<String, Double> map = getMapFromMap(probTransitions, stateFrom);
		incMapEntry(map, stateTo, val);
	}

	public void incFirstStateCount(String state, double val) {
		incMapEntry(probFirstState, state, val);
	}

	private void incMapEntry(HashMap<String, Double> map, String key, double val) {
		Double prevVal = map.get(key);
		if (prevVal == null)
			prevVal = val;
		else
			prevVal = prevVal + val;
		map.put(key, prevVal);
	}

	private HashMap<String, Double> getMapFromMap(
			HashMap<String, HashMap<String, Double>> map, String key) {
		HashMap<String, Double> retMap = map.get(key);
		if (retMap == null) {
			retMap = new HashMap<String, Double>();
			map.put(key, retMap);
		}

		return retMap;
	}

	private void fillDistribution(String[] keyAndValueSeq,
			HashMap<String, Double> map) {
		for (int idx = 1; idx < keyAndValueSeq.length; idx += 2) {
			String key = keyAndValueSeq[idx];
			Double val = Double.parseDouble(keyAndValueSeq[idx + 1]);
			map.put(key, val);
		}
	}

	public Iterable<Entry<String, Double>> getFirstStateIterable() {
		return probFirstState.entrySet();
	}

	public Iterable<Entry<String, HashMap<String, Double>>> getEmissionIterable() {
		return probEmissions.entrySet();
	}

	public Iterable<Entry<String, HashMap<String, Double>>> getTransitionIterable() {
		return probTransitions.entrySet();
	}

}
