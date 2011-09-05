package br.pucrio.inf.learn.structlearning.discriminative.application.sequence.evaluation;

import br.pucrio.inf.learn.structlearning.discriminative.evaluation.TypedEntity;
import br.pucrio.inf.learn.util.HashCodeUtil;

/**
 * Encode the label and the index of a labeled token.
 * 
 * @author eraldo
 * 
 */
public class LabeledToken implements TypedEntity {

	/**
	 * Token label.
	 */
	private String label;

	/**
	 * Token offset within its sequence.
	 */
	private int offset;

	/**
	 * Create a new labeled token entity with the given information.
	 * 
	 * @param label
	 * @param offset
	 */
	public LabeledToken(String label, int offset) {
		this.label = label;
		this.offset = offset;
	}

	@Override
	public boolean equals(Object obj) {
		if (getClass() != obj.getClass())
			return false;
		LabeledToken lt = (LabeledToken) obj;
		return offset == lt.offset && label.equals(lt.label);
	}

	@Override
	public int compareTo(TypedEntity o) {
		LabeledToken lt = (LabeledToken) o;
		if (offset != lt.offset) {
			if (offset < lt.offset)
				return -1;
			return 1;
		}
		return label.compareTo(lt.label);
	}

	@Override
	public String getType() {
		return label;
	}

	/**
	 * Return the offset of this token within its sequence.
	 * 
	 * @return
	 */
	public int getOffset() {
		return offset;
	}

	@Override
	public int hashCode() {
		return HashCodeUtil.hash(
				HashCodeUtil.hash(HashCodeUtil.SEED, label.hashCode()), offset);
	}

}
