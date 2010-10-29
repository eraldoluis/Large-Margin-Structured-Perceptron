package tagger.data;

import java.util.Collection;
import java.util.HashMap;
import java.util.Vector;

import org.apache.log4j.Logger;

/**
 * Encode a set of feature-values as integers to improve algorithm performance.
 * 
 * Provide methods to add new values, obtain the code of a value, obtain the
 * string value of a code, etc.
 * 
 * @author jordi
 * 
 */
public class FeatureValueEncoding {

	static Logger logger = Logger.getLogger(FeatureValueEncoding.class);

	public static final int modeList = 0;
	public static final int modeFeatures = 1;

	private HashMap<String, Integer> valuesMapping;

	// ID (int) to LABLE (String) map
	private Vector<String> values;

	public int size() {
		return values.size();
	}

	public String get(int pos) {
		return values.get(pos);
	}

	// Constructor
	public FeatureValueEncoding() {
		valuesMapping = new HashMap<String, Integer>();
		values = new Vector<String>();
	}

	public int putString(String value) {
		if (value == null)
			throw new NullPointerException("You can not insert a null value.");

		Integer code = valuesMapping.get(value);
		if (code == null) {
			code = values.size();
			values.add(value);
			valuesMapping.put(value, code);
		}

		return code;
	}

	public String getStringByCode(int code) {
		return values.get(code);
	}

	public int getCodeByString(String value) {
		Integer code = valuesMapping.get(value);
		if (code == null)
			return -1;
		return code;
	}

	public Collection<Integer> getCollectionOfSymbols() {
		return valuesMapping.values();
	}
}
