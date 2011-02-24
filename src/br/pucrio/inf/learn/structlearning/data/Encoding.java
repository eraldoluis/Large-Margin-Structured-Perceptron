package br.pucrio.inf.learn.structlearning.data;

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
 * Encode a dictionary of arbitrarily-typed values as integer values.
 * 
 * @author eraldo
 * 
 * @param <ValueType>
 */
public abstract class Encoding<ValueType> {

	private HashMap<ValueType, Integer> mapFromValueToCode;

	private Vector<ValueType> mapFromCodeToValue;

	public Encoding() {
		mapFromValueToCode = new HashMap<ValueType, Integer>();
		mapFromCodeToValue = new Vector<ValueType>();
	}

	public Encoding(InputStream is) throws IOException {
		this();
		load(is);
	}

	public Encoding(BufferedReader reader) throws IOException {
		this();
		load(reader);
	}

	public Encoding(String fileName) throws IOException {
		this();
		load(fileName);
	}

	public int size() {
		return mapFromCodeToValue.size();
	}

	public int putValue(ValueType value) {
		if (value == null)
			throw new NullPointerException("You can not insert a null feature.");

		Integer code = mapFromValueToCode.get(value);
		if (code == null) {
			code = mapFromCodeToValue.size();
			mapFromCodeToValue.add(value);
			mapFromValueToCode.put(value, code);
		}

		return code;
	}

	public ValueType getFeatureByCode(int code) {
		return mapFromCodeToValue.get(code);
	}

	public int getCodeByFeature(ValueType value) {
		Integer code = mapFromValueToCode.get(value);
		if (code == null)
			return -1;
		return code;
	}

	public Collection<Integer> getCodes() {
		return mapFromValueToCode.values();
	}

	public Collection<ValueType> getOrderedValues() {
		return mapFromCodeToValue;
	}

	public abstract void load(BufferedReader reader) throws IOException;

	public abstract void save(PrintStream ps);

	public void load(InputStream is) throws IOException {
		BufferedReader reader = new BufferedReader(new InputStreamReader(is));
		load(reader);
	}

	public void load(String fileName) throws IOException {
		BufferedReader reader = new BufferedReader(new FileReader(fileName));
		load(reader);
		reader.close();
	}

	public void save(OutputStream os) {
		PrintStream ps = new PrintStream(os);
		save(ps);
		ps.flush();
	}

	public void save(String fileName) throws FileNotFoundException {
		PrintStream ps = new PrintStream(fileName);
		save(ps);
		ps.close();
	}
}
