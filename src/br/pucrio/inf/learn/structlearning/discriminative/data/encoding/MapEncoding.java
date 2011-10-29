package br.pucrio.inf.learn.structlearning.discriminative.data.encoding;

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
import java.util.Map;
import java.util.Vector;

/**
 * Encode a dictionary of arbitrarily-typed values as integer values.
 * 
 * Store the dictionary using a HashMap and an array.
 * 
 * @author eraldo
 * 
 * @param <ValueType>
 */
public abstract class MapEncoding<ValueType> implements
		FeatureEncoding<ValueType> {

	/**
	 * Map from values to codes.
	 */
	protected Map<ValueType, Integer> mapFromValueToCode;

	/**
	 * Map from codes to values.
	 */
	protected Vector<ValueType> mapFromCodeToValue;

	/**
	 * If <code>true</code>, the user cannot add more values to this encoding.
	 */
	protected boolean readOnly;

	/**
	 * Default constructor. Create an empty encoding.
	 */
	public MapEncoding() {
		mapFromValueToCode = new HashMap<ValueType, Integer>();
		mapFromCodeToValue = new Vector<ValueType>();
		readOnly = false;
	}

	/**
	 * Create an encoding with the given values.
	 * 
	 * @param values
	 */
	public MapEncoding(ValueType[] values) {
		this();
		for (ValueType val : values)
			put(val);
		readOnly = true;
	}

	/**
	 * Create an encoding and add the values in <code>is</code>. The encoding is
	 * set to read-only. The user can change this calling the method
	 * <code>setReadOnly</code>.
	 * 
	 * @param is
	 * @throws IOException
	 */
	public MapEncoding(InputStream is) throws IOException {
		this();
		load(is);
		readOnly = true;
	}

	/**
	 * Create an encoding and add the values in <code>reader</code>. The
	 * encoding is set to read-only. The user can change this calling the method
	 * <code>setReadOnly</code>.
	 * 
	 * @param reader
	 * @throws IOException
	 */
	public MapEncoding(BufferedReader reader) throws IOException {
		this();
		load(reader);
		readOnly = true;
	}

	/**
	 * Create an encoding and add the values read from the file named
	 * <code>fileName</code>. The encoding is set to read-only. The user can
	 * change this calling the method <code>setReadOnly</code>.
	 * 
	 * @param fileName
	 * @throws IOException
	 */
	public MapEncoding(String fileName) throws IOException {
		this();
		load(fileName);
		readOnly = true;
	}

	/**
	 * Return the number of values in this encoding.
	 * 
	 * @return
	 */
	public int size() {
		return mapFromCodeToValue.size();
	}

	/**
	 * Inset a value in this encoding and return its code. If the value already
	 * exists in the encoding, only return its code. If <code>readOnly</code> is
	 * <code>true</code> and the given value is not present in the encoding, do
	 * not add it and return <code>UNSEEN_VALUE_CODE</code>.
	 * 
	 * @param value
	 * @return
	 */
	public int put(ValueType value) {
		if (value == null)
			throw new NullPointerException("You can not insert a null value.");

		Integer code = mapFromValueToCode.get(value);
		if (code == null) {
			if (readOnly)
				return UNSEEN_VALUE_CODE;
			code = mapFromCodeToValue.size();
			mapFromCodeToValue.add(value);
			mapFromValueToCode.put(value, code);
		}

		return code;
	}

	/**
	 * Return the value associated with the given code. If this code does not
	 * exist, an exception is launched.
	 * 
	 * @param code
	 * @return
	 */
	public ValueType getValueByCode(int code) {
		if (code < 0)
			return null;
		return mapFromCodeToValue.get(code);
	}

	/**
	 * Return the code of the given value. If the value is not in the encoding,
	 * return <code>UNSEEN_VALUE_CODE</code>.
	 * 
	 * @param value
	 * @return
	 */
	public int getCodeByValue(ValueType value) {
		Integer code = mapFromValueToCode.get(value);
		if (code == null)
			return UNSEEN_VALUE_CODE;
		return code;
	}

	/**
	 * Return a collection with all codes used in this encoding.
	 * 
	 * @return
	 */
	public Collection<Integer> getCodes() {
		return mapFromValueToCode.values();
	}

	/**
	 * Return a collection with all the values in this encoding.
	 * 
	 * @return
	 */
	public Collection<ValueType> getValues() {
		return mapFromCodeToValue;
	}

	public void setReadOnly(boolean b) {
		readOnly = b;
	}

	/**
	 * Read values from <code>is</code> and add them to this encoding.
	 * 
	 * @param is
	 * @throws IOException
	 */
	public void load(InputStream is) throws IOException {
		BufferedReader reader = new BufferedReader(new InputStreamReader(is));
		load(reader);
	}

	/**
	 * Read values from the file named <code>fileName</code> and add them to
	 * this encoding.
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
	 * Write the values of this encoding to <code>os</code>.
	 * 
	 * @param os
	 */
	public void save(OutputStream os) {
		PrintStream ps = new PrintStream(os);
		save(ps);
		ps.flush();
	}

	/**
	 * Write the values of this encoding to the file named <code>fileName</code>
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
	 * Read values from <code>reader</code> and add them to this encoding.
	 * Concrete sub-classes of <code>Encoding</code> must implement this method
	 * to provide the required funcionality.
	 * 
	 * @param reader
	 * @throws IOException
	 */
	public abstract void load(BufferedReader reader) throws IOException;

	/**
	 * Write the values of this encoding to <code>ps</code>. Concrete
	 * sub-classes of <code>Encoding</code> must implement this method to
	 * provide the required funcionality.
	 * 
	 * @param ps
	 */
	public abstract void save(PrintStream ps);
}
