package br.pucrio.inf.learn.structlearning.data;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.io.PrintStream;
import java.util.Collection;

/**
 * Encode a dictionary of feature values as integer values (codes).
 * 
 * The codes must be within the interval [0,<size>], where <size> is the number
 * of values in the dictionary.
 * 
 * @author eraldo
 * 
 * @param <ValueType>
 */
public interface FeatureEncoding<ValueType> {

	/**
	 * Special (invalid) code to unseen values. This code is returned when the
	 * user asks for the code of an previously unseen value (
	 * <code>getCodeByValue</code>) or tries to put a new symbol in a read-only
	 * encoding object (<code>put</code>).
	 */
	public static final int UNSEEN_VALUE_CODE = -1;

	/**
	 * Return the number of values in this encoding.
	 * 
	 * @return
	 */
	public int size();

	/**
	 * Insert a value in this encoding and return its code.
	 * 
	 * If the value already exists in the encoding, only return its code. If
	 * <code>readOnly</code> is <code>true</code> and the given value is not
	 * present in the encoding, do not add it and return
	 * <code>UNSEEN_VALUE_CODE</code>.
	 * 
	 * @param value
	 * @return
	 */
	public int put(ValueType value);

	/**
	 * Return the value associated with the given code.
	 * 
	 * If this code does not exist, null is returned.
	 * 
	 * @param code
	 * @return
	 */
	public ValueType getValueByCode(int code);

	/**
	 * Return the code of the given value.
	 * 
	 * If the value is not in the encoding, return
	 * <code>UNSEEN_VALUE_CODE</code>.
	 * 
	 * @param value
	 * @return
	 */
	public int getCodeByValue(ValueType value);

	/**
	 * Return a collection with all codes used in this encoding.
	 * 
	 * @return
	 */
	public Collection<Integer> getCodes();

	/**
	 * Return a collection with all the values in this encoding.
	 * 
	 * @return
	 */
	public Collection<ValueType> getValues();

	/**
	 * Allows the user to freeze this dictionary at the current state.
	 * 
	 * If the user calls this method with a <code>true</code> value, any
	 * subsequent call to the method put will not add the value if this value is
	 * not present.
	 * 
	 * @param b
	 */
	public void setReadOnly(boolean b);

	/**
	 * Read values from <code>is</code> and add them to this encoding.
	 * 
	 * @param is
	 * @throws IOException
	 */
	public void load(InputStream is) throws IOException;

	/**
	 * Read values from the file named <code>fileName</code> and add them to
	 * this encoding.
	 * 
	 * @param fileName
	 * @throws IOException
	 */
	public void load(String fileName) throws IOException;

	/**
	 * Write the values of this encoding to <code>os</code>.
	 * 
	 * @param os
	 */
	public void save(OutputStream os);

	/**
	 * Write the values of this encoding to the file named <code>fileName</code>
	 * 
	 * @param fileName
	 * @throws FileNotFoundException
	 */
	public void save(String fileName) throws FileNotFoundException;

	/**
	 * Read values from <code>reader</code> and add them to this encoding.
	 * Concrete sub-classes of <code>Encoding</code> must implement this method
	 * to provide the required funcionality.
	 * 
	 * @param reader
	 * @throws IOException
	 */
	public void load(BufferedReader reader) throws IOException;

	/**
	 * Write the values of this encoding to <code>ps</code>. Concrete
	 * sub-classes of <code>Encoding</code> must implement this method to
	 * provide the required funcionality.
	 * 
	 * @param ps
	 */
	public void save(PrintStream ps);
}
