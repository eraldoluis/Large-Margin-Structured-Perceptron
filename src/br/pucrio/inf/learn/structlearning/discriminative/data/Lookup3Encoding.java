package br.pucrio.inf.learn.structlearning.discriminative.data;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.io.PrintStream;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Random;

import br.pucrio.inf.learn.util.Lookup3Hash;

/**
 * A string feature encoding based on the Lookup3 hashing function.
 * 
 * This encoding is fixed, i.e., the user gives the desired size and this class
 * uses the Lookup3 hashing function to generate the codes for each value on
 * demand.
 * 
 * @author eraldof
 * 
 */
public class Lookup3Encoding implements FeatureEncoding<String> {

	/**
	 * This is the only parameter of the Murmur3 hashing function.
	 * 
	 * By using different seeds, one can create different hashing functions.
	 */
	private final int seed;

	/**
	 * Number of possible (and fixed) codes of this encoding.
	 */
	private final int size;

	/**
	 * Create an encoding with the given size (number of possible codes) and the
	 * given seed.
	 * 
	 * @param size
	 * @param seed
	 */
	public Lookup3Encoding(int size, int seed) {
		this.size = size;
		this.seed = seed;
	}

	/**
	 * Create an encoding with the given size (number of possible codes) and a
	 * random seed.
	 * 
	 * @param size
	 */
	public Lookup3Encoding(int size) {
		this(size, new Random().nextInt());
	}

	@Override
	public int size() {
		return size;
	}

	@Override
	public int put(String value) {
		return Math.abs(Lookup3Hash.hash32(value.getBytes(), seed)) % size;
	}

	@Override
	public String getValueByCode(int code) {
		return null;
	}

	@Override
	public int getCodeByValue(String value) {
		return put(value);
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
