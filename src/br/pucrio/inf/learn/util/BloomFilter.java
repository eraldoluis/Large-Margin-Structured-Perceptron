/**
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

package br.pucrio.inf.learn.util;

import java.io.Serializable;
import java.nio.charset.Charset;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.BitSet;
import java.util.Collection;

/**
 * Implementation of a Bloom-filter, as described here:
 * http://en.wikipedia.org/wiki/Bloom_filter
 * 
 * Inspired by the SimpleBloomFilter-class written by Ian Clarke. This
 * implementation provides a more evenly distributed Hash-function by using a
 * proper digest instead of the Java RNG. Many of the changes were proposed in
 * comments in his blog:
 * http://blog.locut.us/2008/01/12/a-decent-stand-alone-java
 * -bloom-filter-implementation/
 * 
 * @param <KeyType>
 *            Object type that is to be inserted into the Bloom filter, e.g.
 *            String or Integer.
 * @author Magnus Skjegstad <magnus@skjegstad.com>
 */
public class BloomFilter<KeyType> implements Serializable {
	/**
	 * Generated UID.
	 */
	private static final long serialVersionUID = -3415797794782171016L;

	/**
	 * Store the basic structure of bits.
	 */
	private BitSet bitset;

	/**
	 * Estimated number of bits per element.
	 */
	private double bitsPerElement;

	/**
	 * Expected (maximum) number of elements to be added.
	 */
	private int expectedNumberOfFilterElements;

	/**
	 * Number of times that the add method was called.
	 */
	private int numberOfInsertions;

	/**
	 * Number of hash functions.
	 */
	private int numberOfHashFunctions;

	/**
	 * Encoding used for storing hash values as strings.
	 */
	static final Charset charset = Charset.forName("UTF-8");

	/**
	 * MD5 gives good enough accuracy in most circumstances. Change to SHA1 if
	 * it's needed.
	 */
	static final String hashName = "MD5";

	/**
	 * Shared digest object.
	 */
	static final MessageDigest digestFunction;

	/**
	 * Load the digest method object.
	 */
	static {
		MessageDigest tmp;
		try {
			tmp = java.security.MessageDigest.getInstance(hashName);
		} catch (NoSuchAlgorithmException e) {
			tmp = null;
		}
		digestFunction = tmp;
	}

	/**
	 * Constructs an empty Bloom filter. The total length of the Bloom filter
	 * will be c*n.
	 * 
	 * @param numberOfBitsPerElement
	 *            is the number of bits used per element.
	 * @param expectedNumberfOfElements
	 *            is the expected number of elements the filter will contain.
	 * @param numberOfHashFunctions
	 *            is the number of hash functions used.
	 */
	public BloomFilter(double numberOfBitsPerElement,
			int expectedNumberfOfElements, int numberOfHashFunctions) {
		this.expectedNumberOfFilterElements = expectedNumberfOfElements;
		this.numberOfHashFunctions = numberOfHashFunctions;
		this.bitsPerElement = numberOfBitsPerElement;
		numberOfInsertions = 0;
		this.bitset = new BitSet((int) Math.ceil(numberOfBitsPerElement
				* expectedNumberfOfElements));
	}

	/**
	 * Constructs an empty Bloom filter. The optimal number of hash functions
	 * (k) is estimated from the total size of the Bloom and the number of
	 * expected elements.
	 * 
	 * @param bitSetSize
	 *            defines how many bits should be used in total for the filter.
	 * @param expectedNumberOElements
	 *            defines the maximum number of elements the filter is expected
	 *            to contain.
	 */
	public BloomFilter(int bitSetSize, int expectedNumberOElements) {
		this(bitSetSize / (double) expectedNumberOElements,
				expectedNumberOElements, (int) Math
						.round((bitSetSize / (double) expectedNumberOElements)
								* Math.log(2.0)));
	}

	/**
	 * Constructs an empty Bloom filter with a given false positive probability.
	 * The number of bits per element and the number of hash functions is
	 * estimated to match the false positive probability.
	 * 
	 * @param falsePositiveProbability
	 *            is the desired false positive probability.
	 * @param expectedNumberOfElements
	 *            is the expected number of elements in the Bloom filter.
	 */
	public BloomFilter(double falsePositiveProbability,
			int expectedNumberOfElements) {
		// c = k / ln(2)
		// k = ceil(-log_2(falseprob))
		this(Math.ceil(-(Math.log(falsePositiveProbability) / Math.log(2)))
				/ Math.log(2), expectedNumberOfElements, (int) Math.ceil(-(Math
				.log(falsePositiveProbability) / Math.log(2))));
	}

	/**
	 * Construct a new Bloom filter based on existing Bloom filter data.
	 * 
	 * @param bitSetSize
	 *            defines how many bits should be used for the filter.
	 * @param expectedNumberOfFilterElements
	 *            defines the maximum number of elements the filter is expected
	 *            to contain.
	 * @param actualNumberOfFilterElements
	 *            specifies how many elements have been inserted into the
	 *            <code>filterData</code> BitSet.
	 * @param filterData
	 *            a BitSet representing an existing Bloom filter.
	 */
	public BloomFilter(int bitSetSize, int expectedNumberOfFilterElements,
			int actualNumberOfFilterElements, BitSet filterData) {
		this(bitSetSize, expectedNumberOfFilterElements);
		this.bitset = filterData;
		this.numberOfInsertions = actualNumberOfFilterElements;
	}

	/**
	 * Generates a digest based on the contents of a String.
	 * 
	 * @param val
	 *            specifies the input data.
	 * @param charset
	 *            specifies the encoding of the input data.
	 * @return digest as long.
	 */
	public static long createHash(String val, Charset charset) {
		return createHash(val.getBytes(charset));
	}

	/**
	 * Generates a digest based on the contents of a String.
	 * 
	 * @param val
	 *            specifies the input data. The encoding is expected to be
	 *            UTF-8.
	 * @return digest as long.
	 */
	public static long createHash(String val) {
		return createHash(val, charset);
	}

	/**
	 * Generates a digest based on the contents of an array of bytes.
	 * 
	 * @param data
	 *            specifies input data.
	 * @return digest as long.
	 */
	public static long createHash(byte[] data) {
		long h = 0;
		byte[] res;

		synchronized (digestFunction) {
			res = digestFunction.digest(data);
		}

		for (int i = 0; i < 4; i++) {
			h <<= 8;
			h |= ((int) res[i]) & 0xFF;
		}
		return h;
	}

	/**
	 * Compares the contents of two instances to see if they are equal.
	 * 
	 * @param obj
	 *            is the object to compare to.
	 * @return True if the contents of the objects are equal.
	 */
	@Override
	public boolean equals(Object obj) {
		if (obj == null) {
			return false;
		}
		if (getClass() != obj.getClass()) {
			return false;
		}
		final BloomFilter<KeyType> other = (BloomFilter<KeyType>) obj;
		if (this.expectedNumberOfFilterElements != other.expectedNumberOfFilterElements) {
			return false;
		}
		if (this.numberOfHashFunctions != other.numberOfHashFunctions) {
			return false;
		}
		if (this.bitset != other.bitset) {
			if (this.bitset == null)
				return false;
			if (!this.bitset.equals(other.bitset))
				return false;
		}
		return true;
	}

	/**
	 * Calculates a hash code for this class.
	 * 
	 * @return hash code representing the contents of an instance of this class.
	 */
	@Override
	public int hashCode() {
		int hash = 7;
		hash = 61 * hash + (this.bitset != null ? this.bitset.hashCode() : 0);
		hash = 61 * hash + this.expectedNumberOfFilterElements;
		hash = 61 * hash + this.numberOfHashFunctions;
		return hash;
	}

	/**
	 * Calculates the expected probability of false positives based on the
	 * number of expected filter elements and the size of the Bloom filter. <br />
	 * <br />
	 * The value returned by this method is the <i>expected</i> rate of false
	 * positives, assuming the number of inserted elements equals the number of
	 * expected elements. If the number of elements in the Bloom filter is less
	 * than the expected value, the true probability of false positives will be
	 * lower.
	 * 
	 * @return expected probability of false positives.
	 */
	public double expectedFalsePositiveProbability() {
		return getFalsePositiveProbability(expectedNumberOfFilterElements);
	}

	/**
	 * Calculate the probability of a false positive given the specified number
	 * of inserted elements.
	 * 
	 * @param numberOfElements
	 *            number of inserted elements.
	 * @return probability of a false positive.
	 */
	public double getFalsePositiveProbability(double numberOfElements) {
		// (1 - e^(-k * n / m)) ^ k
		return Math.pow(
				(1 - Math.exp(-numberOfHashFunctions
						* (double) numberOfElements / (double) bitset.size())),
				numberOfHashFunctions);
	}

	/**
	 * Get the current probability of a false positive. The probability is
	 * calculated from the size of the Bloom filter and the current number of
	 * elements added to it.
	 * 
	 * @return probability of false positives.
	 */
	public double getFalsePositiveProbability() {
		return getFalsePositiveProbability(numberOfInsertions);
	}

	/**
	 * Returns the number of hash functions.
	 * 
	 * @return optimal k.
	 */
	public int getNumberOfHashFunctions() {
		return numberOfHashFunctions;
	}

	/**
	 * Sets all bits to false in the Bloom filter.
	 */
	public void clear() {
		bitset.clear();
		numberOfInsertions = 0;
	}

	/**
	 * Adds an object to the Bloom filter. The output from the object's
	 * toString() method is used as input to the hash functions.
	 * 
	 * @param element
	 *            is an element to register in the Bloom filter.
	 */
	public void add(KeyType element) {
		long hash;
		String valString = element.toString();
		for (int x = 0; x < numberOfHashFunctions; x++) {
			hash = createHash(valString + Integer.toString(x));
			hash = hash % (long) bitset.size();
			bitset.set(Math.abs((int) hash), true);
		}
		numberOfInsertions++;
	}

	/**
	 * Adds all elements from a Collection to the Bloom filter.
	 * 
	 * @param c
	 *            Collection of elements.
	 */
	public void addAll(Collection<? extends KeyType> c) {
		for (KeyType element : c)
			add(element);
	}

	/**
	 * Returns true if the element could have been inserted into the Bloom
	 * filter. Use getFalsePositiveProbability() to calculate the probability of
	 * this being correct.
	 * 
	 * @param element
	 *            element to check.
	 * @return true if the element could have been inserted into the Bloom
	 *         filter.
	 */
	public boolean contains(KeyType element) {
		long hash;
		String valString = element.toString();
		for (int x = 0; x < numberOfHashFunctions; x++) {
			hash = createHash(valString + Integer.toString(x));
			hash = hash % (long) bitset.size();
			if (!bitset.get(Math.abs((int) hash)))
				return false;
		}
		return true;
	}

	/**
	 * Returns true if all the elements of a Collection could have been inserted
	 * into the Bloom filter. Use getFalsePositiveProbability() to calculate the
	 * probability of this being correct.
	 * 
	 * @param c
	 *            elements to check.
	 * @return true if all the elements in c could have been inserted into the
	 *         Bloom filter.
	 */
	public boolean containsAll(Collection<? extends KeyType> c) {
		for (KeyType element : c)
			if (!contains(element))
				return false;
		return true;
	}

	/**
	 * Read a single bit from the Bloom filter.
	 * 
	 * @param bit
	 *            the bit to read.
	 * @return true if the bit is set, false if it is not.
	 */
	public boolean getBit(int bit) {
		return bitset.get(bit);
	}

	/**
	 * Set a single bit in the Bloom filter.
	 * 
	 * @param bit
	 *            is the bit to set.
	 * @param value
	 *            If true, the bit is set. If false, the bit is cleared.
	 */
	public void setBit(int bit, boolean value) {
		bitset.set(bit, value);
	}

	/**
	 * Return the bit set used to store the Bloom filter.
	 * 
	 * @return bit set representing the Bloom filter.
	 */
	public BitSet getBitSet() {
		return bitset;
	}

	/**
	 * Returns the number of bits in the Bloom filter. Use count() to retrieve
	 * the number of inserted elements.
	 * 
	 * @return the size of the bitset used by the Bloom filter.
	 */
	public int size() {
		return this.bitset.size();
	}

	/**
	 * Returns the number of elements added to the Bloom filter after it was
	 * constructed or after clear() was called.
	 * 
	 * @return number of elements added to the Bloom filter.
	 */
	public int count() {
		return this.numberOfInsertions;
	}

	/**
	 * Returns the expected number of elements to be inserted into the filter.
	 * This value is the same value as the one passed to the constructor.
	 * 
	 * @return expected number of elements.
	 */
	public int getExpectedNumberOfElements() {
		return expectedNumberOfFilterElements;
	}

	/**
	 * Get expected number of bits per element when the Bloom filter is full.
	 * This value is set by the constructor when the Bloom filter is created.
	 * See also getBitsPerElement().
	 * 
	 * @return expected number of bits per element.
	 */
	public double getExpectedBitsPerElement() {
		return this.bitsPerElement;
	}

	/**
	 * Get the expected number of bits per element based on the number of
	 * insertions performed up to now and the length of the Bloom filter.
	 * 
	 * @return number of bits per element.
	 */
	public double getBitsPerElement() {
		return this.bitset.size() / (double) numberOfInsertions;
	}
}
