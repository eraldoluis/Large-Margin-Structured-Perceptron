package br.pucrio.inf.learn.structlearning.generative.data;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.io.PrintStream;
import java.util.Collection;
import java.util.HashMap;
import java.util.Vector;

/**
 * Encode a set of feature-values as integers to improve algorithm performance.
 * 
 * Provide methods to add new values, obtain the code of a value, obtain the
 * string value of a code, etc.
 * 
 * @author eraldof
 * 
 */
public class FeatureValueEncoding {

	/**
	 * Reserved label for unseen labels when using read-only mapping.
	 */
	public static final String UNSEEN_LABEL = "__UNSEENLABEL__";

	/**
	 * Reserved code for unseen labels when using read-only mapping.
	 */
	public static final int UNSEEN_CODE = Integer.MAX_VALUE;

	/**
	 * Label (String) to code (Integer) mapping.
	 */
	private HashMap<String, Integer> mapFromLabelToCode;

	/**
	 * Code (Integer) to label (String) mapping.
	 */
	private Vector<String> mapFromCodeToLabel;

	/**
	 * Indicate if this mapping is read-only.
	 */
	private boolean readOnly;

	/**
	 * Default constructor. Create an empty mapping.
	 */
	public FeatureValueEncoding() {
		this.readOnly = false;
		mapFromLabelToCode = new HashMap<String, Integer>();
		mapFromCodeToLabel = new Vector<String>();
	}

	/**
	 * Used only internally to construct read-only mappings.
	 * 
	 * @param readOnly
	 */
	private FeatureValueEncoding(boolean readOnly) {
		this.readOnly = readOnly;
		mapFromLabelToCode = new HashMap<String, Integer>();
		mapFromCodeToLabel = new Vector<String>();
	}

	/**
	 * Create a new encoding by loading the given input stream.
	 * 
	 * @param is
	 * @param readOnly
	 *            indicate if new labels can be added to the mapping.
	 * @throws IOException
	 */
	public FeatureValueEncoding(InputStream is, boolean readOnly)
			throws IOException {
		this(readOnly);
		load(is);
	}

	/**
	 * Create a new encoding by loading the given input stream.
	 * 
	 * @param reader
	 * @param readOnly
	 *            indicate if new labels can be added to the mapping.
	 * @throws IOException
	 */
	public FeatureValueEncoding(BufferedReader reader, boolean readOnly)
			throws IOException {
		this(readOnly);
		load(reader);
	}

	/**
	 * Create a new encoding by loading the given file.
	 * 
	 * @param fileName
	 * @param readOnly
	 *            indicate if new labels can be added to the mapping.
	 * @throws IOException
	 */
	public FeatureValueEncoding(String fileName, boolean readOnly)
			throws IOException {
		this(readOnly);
		load(fileName);
	}

	/**
	 * @return the number of values encoded within this mapping.
	 */
	public int size() {
		return mapFromCodeToLabel.size();
	}

	public boolean isReadOnly() {
		return readOnly;
	}

	public void setReadOnly(boolean val) {
		readOnly = val;
	}

	/**
	 * Put the given label in this mapping and return its code. If the label
	 * already is in the mapping, just return its code.
	 * 
	 * @param label
	 *            a label to put in this mapping or to retrieve its code.
	 * 
	 * @return the code of the given label. It may be a new code (for a new
	 *         label) or a previously existing code (for a label that already
	 *         was in the mapping).
	 */
	public int putString(String label) {
		if (label == null)
			throw new NullPointerException("You can not insert a null value.");

		Integer code = mapFromLabelToCode.get(label);
		if (code == null) {
			// If this is a read-only mapping, return the unseen code.
			if (readOnly || label.equals(UNSEEN_LABEL))
				return UNSEEN_CODE;

			code = mapFromCodeToLabel.size();
			mapFromCodeToLabel.add(label);
			mapFromLabelToCode.put(label, code);
		}

		return code;
	}

	/**
	 * Return the label corresponding to the given code.
	 * 
	 * @param code
	 *            a code within this mapping.
	 * @return the label corresponding to the given code.
	 */
	public String getLabelByCode(int code) {
		if (code == UNSEEN_CODE)
			return UNSEEN_LABEL;
		return mapFromCodeToLabel.get(code);
	}

	/**
	 * Return the code corresponding to the given label.
	 * 
	 * @param label
	 *            a label within this mapping.
	 * @return the code corresponding to the given label or -1 if the label is
	 *         not present in this mapping.
	 */
	public int getCodeByLabel(String label) {
		Integer code = mapFromLabelToCode.get(label);
		if (code == null) {
			if (readOnly || label.equals(UNSEEN_LABEL))
				return UNSEEN_CODE;
			return -1;
		}
		return code;
	}

	/**
	 * Return a collection with all the labels in this mapping.
	 * 
	 * The collection doest not contain the special unseen label.
	 * 
	 * @return
	 */
	public Collection<Integer> getCollectionOfLabels() {
		return mapFromLabelToCode.values();
	}

	/**
	 * Load an encoding from the given input stream. Clear the current encoding
	 * before loading the new one.
	 * 
	 * @param is
	 * @throws IOException
	 */
	public void load(InputStream is) throws IOException {
		BufferedReader reader = new BufferedReader(new InputStreamReader(is));
		load(reader);
	}

	/**
	 * Load an encoding from the given file. Clear the current encoding before
	 * loading the new one.
	 * 
	 * @param fileName
	 * @throws IOException
	 */
	public void load(String fileName) throws IOException {
		BufferedReader reader = new BufferedReader(new FileReader(fileName));
		load(reader);
		reader.close();
	}

	/**
	 * Load an encoding from the given reader. Clear the current encoding before
	 * loading the new one.
	 * 
	 * @param reader
	 * @throws IOException
	 */
	public void load(BufferedReader reader) throws IOException {
		// Clear the previous values.
		mapFromCodeToLabel.clear();
		mapFromLabelToCode.clear();

		String label;
		while ((label = reader.readLine()) != null) {
			int code = mapFromCodeToLabel.size();
			mapFromCodeToLabel.add(label);
			mapFromLabelToCode.put(label, code);
		}
	}

	public void loadTwoPasses(BufferedReader reader) throws IOException {
		// Clear the previous values.
		mapFromCodeToLabel.clear();
		mapFromLabelToCode.clear();

		int numberOfLabels = 0;
		String label;
		while ((label = reader.readLine()) != null)
			++numberOfLabels;

		mapFromCodeToLabel.setSize(numberOfLabels);
		mapFromCodeToLabel.setSize(0);
		while ((label = reader.readLine()) != null) {
			int code = mapFromCodeToLabel.size();
			mapFromCodeToLabel.add(label);
			mapFromLabelToCode.put(label, code);
		}
	}

	/**
	 * Save this encoding in the given output stream. The given stream is not
	 * closed in this method.
	 * 
	 * @param os
	 */
	public void save(OutputStream os) {
		PrintStream ps = new PrintStream(os);
		save(ps);
		ps.flush();
	}

	/**
	 * Save this encoding in the given file.
	 * 
	 * @param fileName
	 * @throws FileNotFoundException
	 */
	public void save(String fileName) throws FileNotFoundException {
		PrintStream ps = new PrintStream(fileName);
		save(ps);
		ps.close();
	}

	/**
	 * Save this encoding in the given print stream. The given stream is not
	 * closed.
	 * 
	 * @param ps
	 */
	public void save(PrintStream ps) {
		for (String label : mapFromCodeToLabel)
			ps.println(label);
		ps.flush();
	}

}
