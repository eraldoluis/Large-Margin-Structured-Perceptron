package br.pucrio.inf.learn.util;

import java.lang.reflect.Array;

/**
 * Collected methods which allow easy implementation of <code>hashCode</code>.
 * 
 * Example use case:
 * 
 * <pre>
 * public int hashCode() {
 * 	int result = HashCodeUtil.SEED;
 * 	// collect the contributions of various fields
 * 	result = HashCodeUtil.hash(result, fPrimitive);
 * 	result = HashCodeUtil.hash(result, fObject);
 * 	result = HashCodeUtil.hash(result, fArray);
 * 	return result;
 * }
 * </pre>
 */
public final class HashCodeUtil {

	/**
	 * List of "good" sizes for a hash table (almost prime). The list is
	 * generated from the number 89 and iteratively doind expansions of the form
	 * 2n+1.
	 */
	public static final int[] GOOD_TABLE_SIZES = { 89, 179, 359, 719, 1439,
			2879, 5759, 11519, 23039, 46079, 92159, 184319, 368639, 737279,
			1474559, 2949119, 5898239, 11796479, 23592959, 47185919, 94371839,
			188743679, 377487359, 754974719, 1509949439 };

	/**
	 * Return a "good" table size that approximates the given number of bits.
	 * That is, the table size will be slightly larger than 2^b, where b is the
	 * given number of bits.
	 * 
	 * @param numberOfBits
	 * @return
	 */
	public static int getGoodSizeForBits(int numberOfBits) {
		int min = 6;
		int max = min - 1 + GOOD_TABLE_SIZES.length;
		if (numberOfBits < min)
			return GOOD_TABLE_SIZES[0];
		if (numberOfBits > max)
			return GOOD_TABLE_SIZES[GOOD_TABLE_SIZES.length - 1];
		return GOOD_TABLE_SIZES[numberOfBits - min];
	}

	/**
	 * An initial value for a <code>hashCode</code>, to which is added
	 * contributions from fields. Using a non-zero value decreases collisons of
	 * <code>hashCode</code> values.
	 */
	public static final int SEED = 1509949439;

	/**
	 * booleans.
	 */
	public static int hash(int aSeed, boolean aBoolean) {
		return firstTerm(aSeed) + (aBoolean ? 1 : 0);
	}

	/**
	 * booleans.
	 */
	public static int hash(boolean aBoolean) {
		return hash(SEED, aBoolean);
	}

	/**
	 * chars.
	 */
	public static int hash(int aSeed, char aChar) {
		return firstTerm(aSeed) + (int) aChar;
	}

	/**
	 * chars.
	 */
	public static int hash(char aChar) {
		return hash(SEED, aChar);
	}

	/**
	 * ints. byte and short are handled by this method, through implicit
	 * conversion.
	 */
	public static int hash(int aSeed, int aInt) {
		return firstTerm(aSeed) + aInt;
	}

	/**
	 * ints.
	 */
	public static int hash(int aInt) {
		return hash(SEED, aInt);
	}

	/**
	 * longs.
	 */
	public static int hash(int aSeed, long aLong) {
		return firstTerm(aSeed) + (int) (aLong ^ (aLong >>> 32));
	}

	/**
	 * longs.
	 */
	public static int hash(long aLong) {
		return hash(SEED, aLong);
	}

	/**
	 * floats.
	 */
	public static int hash(int aSeed, float aFloat) {
		return hash(aSeed, Float.floatToIntBits(aFloat));
	}

	/**
	 * floats.
	 */
	public static int hash(float aFloat) {
		return hash(SEED, aFloat);
	}

	/**
	 * doubles.
	 */
	public static int hash(int aSeed, double aDouble) {
		return hash(aSeed, Double.doubleToLongBits(aDouble));
	}

	/**
	 * doubles.
	 */
	public static int hash(double aDouble) {
		return hash(SEED, aDouble);
	}

	/**
	 * <code>aObject</code> is a possibly-null object field, and possibly an
	 * array.
	 * 
	 * If <code>aObject</code> is an array, then each element may be a primitive
	 * or an object.
	 */
	public static int hash(int aSeed, Object aObject) {
		int result = aSeed;
		if (aObject == null) {
			result = hash(result, 0);
		} else if (!isArray(aObject)) {
			result = hash(result, aObject.hashCode());
		} else {
			int length = Array.getLength(aObject);
			for (int idx = 0; idx < length; ++idx) {
				Object item = Array.get(aObject, idx);
				// recursive call!
				result = hash(result, item);
			}
		}
		return result;
	}

	/**
	 * <code>aObject</code> is a possibly-null object field, and possibly an
	 * array.
	 * 
	 * If <code>aObject</code> is an array, then each element may be a primitive
	 * or an object.
	 */
	public static int hash(Object aObject) {
		return hash(SEED, aObject);
	}

	private static final int fODD_PRIME_NUMBER = 37;

	private static int firstTerm(int aSeed) {
		return fODD_PRIME_NUMBER * aSeed;
	}

	/**
	 * Return <code>true</code> if the given object is an array.
	 * 
	 * @param aObject
	 * @return
	 */
	private static boolean isArray(Object aObject) {
		return aObject.getClass().isArray();
	}

}
