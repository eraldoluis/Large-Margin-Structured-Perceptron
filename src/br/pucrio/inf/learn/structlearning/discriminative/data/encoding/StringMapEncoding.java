package br.pucrio.inf.learn.structlearning.discriminative.data.encoding;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.PrintStream;
import java.util.HashSet;

/**
 * Encoding for string values.
 * 
 * @author eraldo
 * 
 */
public class StringMapEncoding extends MapEncoding<String> {

	/**
	 * Disconsider feature values with low frequency. This only works if each
	 * feature value comes with its frequency in the input file (separed by
	 * space chars).
	 */
	private int minFrequency;

	public StringMapEncoding() {
		super();
	}

	public StringMapEncoding(BufferedReader reader) throws IOException {
		super(reader);
	}

	public StringMapEncoding(InputStream is) throws IOException {
		super(is);
	}

	public StringMapEncoding(String fileName) throws IOException {
		super(fileName);
	}

	/**
	 * Load the feature values from the given file but filter out low frequent
	 * values. Each line in the input file must contain two values separated by
	 * space characters. The first value is the feature value itself and the
	 * second is its frequency count.
	 * 
	 * @param fileName
	 * @param minFrequency
	 * @throws IOException
	 */
	public StringMapEncoding(String fileName, int minFrequency)
			throws IOException {
		super();
		this.minFrequency = minFrequency;
		load(fileName);
		setReadOnly(true);
	}

	public StringMapEncoding(String[] values) {
		super(values);
	}

	@Override
	public void load(BufferedReader reader) throws IOException {
		String line;
		while ((line = reader.readLine()) != null) {
			String[] vals = line.split("\\s");
			if (vals.length == 1)
				put(vals[0]);
			else {
				int freq = Integer.parseInt(vals[1]);
				if (freq >= minFrequency)
					put(vals[0]);
			}
		}
	}

	/**
	 * Only read the size of the encoding, do not store its values.
	 * 
	 * @param fileName
	 * @throws NumberFormatException
	 * @throws IOException
	 */
	public int loadSize(String fileName) throws NumberFormatException,
			IOException {
		BufferedReader reader = new BufferedReader(new FileReader(fileName));
		HashSet<String> valueSet = new HashSet<String>();
		String line;
		while ((line = reader.readLine()) != null) {
			String[] vals = line.split("\\s");
			if (vals.length == 1)
				valueSet.add(vals[0]);
			else {
				int freq = Integer.parseInt(vals[1]);
				if (freq >= minFrequency)
					valueSet.add(vals[0]);
			}
		}
		reader.close();
		return valueSet.size();
	}

	@Override
	public void save(PrintStream ps) {
		for (String val : getValues())
			ps.println(val);
	}

}
