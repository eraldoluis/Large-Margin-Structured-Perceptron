package br.pucrio.inf.learn.util.maxbranching;

import br.pucrio.inf.learn.util.HashCodeUtil;

/**
 * Simple edge representation.
 * 
 * @author eraldo
 * 
 */
public class SimpleWeightedEdge {

	/**
	 * Outgoing node.
	 */
	public final int from;

	/**
	 * Incoming node.
	 */
	public final int to;

	/**
	 * Edge weight.
	 */
	public double weight;

	/**
	 * Constructor.
	 * 
	 * @param from
	 * @param to
	 * @param weight
	 */
	public SimpleWeightedEdge(int from, int to, double weight) {
		this.from = from;
		this.to = to;
		this.weight = weight;
	}

	@Override
	public boolean equals(Object obj) {
		if (!(obj instanceof SimpleWeightedEdge))
			return false;
		SimpleWeightedEdge other = (SimpleWeightedEdge) obj;
		return from == other.from && to == other.to;
	}

	@Override
	public int hashCode() {
		return HashCodeUtil.hash(from, to);
	}

	@Override
	public String toString() {
		return String.format("(%d,%d,%f)", from, to, weight);
	}

}
