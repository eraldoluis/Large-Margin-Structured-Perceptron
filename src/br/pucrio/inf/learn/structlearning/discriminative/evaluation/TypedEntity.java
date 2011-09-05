package br.pucrio.inf.learn.structlearning.discriminative.evaluation;

/**
 * Represent a typed entity that is encoded in an input/output structure. The
 * user of this interface must implement the methods:
 * 
 * <code>getType()</code>: returns the type of the entity.
 * 
 * <code>equals()</code> : used by the <code>EntityF1Evaluation</code>, and any
 * other additional evaluation class, to count how many correct and incorrect
 * entities were identified.
 * 
 * <code>compareTo()</code>: used to maintain some sorted data structures for
 * efficiency matters.
 * 
 * <code>hashCode()</code>: used to store entities in hash tables.
 * 
 * @author eraldof
 * 
 */
public interface TypedEntity extends Comparable<TypedEntity> {

	/**
	 * Return the type of this entity.
	 * 
	 * @return
	 */
	public String getType();

}
