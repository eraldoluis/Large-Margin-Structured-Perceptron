package br.pucrio.inf.learn.structlearning.data;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.io.PrintStream;
import java.util.ArrayList;
import java.util.Collection;

/**
 * A string feature encoding based on the default Java hashing function (
 * <code>hashCode</code> method).
 * 
 * This encoding is fixed, i.e., the user gives the desired size and this class
 * uses the Murmur3 hashing function to generate the codes for each value on
 * demand.
 * 
 * @author eraldof
 * 
 */
public class JavaHashCodeEncoding implements FeatureEncoding<String> {

	/**
	 * Number of possible (and fixed) codes of this encoding.
	 */
	private final int size;

	/**
	 * Create an encoding with the given size (number of possible codes).
	 * 
	 * @param size
	 */
	public JavaHashCodeEncoding(int size) {
		this.size = size;
	}

	@Override
	public int size() {
		return size;
	}

	@Override
	public int put(String value) {
		return value.hashCode() % size;
	}

	@Override
	public String getValueByCode(int code) {
		return null;
	}

	@Override
	public int getCodeByValue(String value) {
		return value.hashCode() % size;
	}

	@Override
	public Collection<Integer> getCodes() {
		// Generate an "artificial" array with all possible codes.
		ArrayList<Integer> codes = new ArrayList<Integer>(size);
		for (int c = 0; c < size; ++c)
			codes.set(c, c);
		return codes;
	}

	@Override
	public Collection<String> getValues() {
		return null;
	}

	@Override
	public void setReadOnly(boolean b) {
		// This Murmur-based encoding is fixed, except for the seed.
	}

	@Override
	public void load(InputStream is) throws IOException {
		// This Murmur-based encoding is fixed, except for the seed.
	}

	@Override
	public void load(String fileName) throws IOException {
		// This Murmur-based encoding is fixed, except for the seed.
	}

	@Override
	public void save(OutputStream os) {
		// This Murmur-based encoding is fixed, except for the seed.
	}

	@Override
	public void save(String fileName) throws FileNotFoundException {
		// This Murmur-based encoding is fixed, except for the seed.
	}

	@Override
	public void load(BufferedReader reader) throws IOException {
		// This Murmur-based encoding is fixed, except for the seed.
	}

	@Override
	public void save(PrintStream ps) {
		// This Murmur-based encoding is fixed, except for the seed.
	}

}
