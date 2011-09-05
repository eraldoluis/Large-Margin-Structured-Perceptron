package br.pucrio.inf.learn.structlearning.discriminative.application.sequence.evaluation;

import br.pucrio.inf.learn.structlearning.discriminative.evaluation.TypedEntity;
import br.pucrio.inf.learn.util.HashCodeUtil;

/**
 * Represent a typed chunk of tokens. It can be used to represent, for instance,
 * phrase chunks, named entities, etc.
 * 
 * @author eraldof
 * 
 */
public class TypedChunk implements TypedEntity {

	/**
	 * The index of the first token of this chunk.
	 */
	public int tokenBeg;

	/**
	 * The index of the last token of this chunk.
	 */
	public int tokenEnd;

	/**
	 * The type of this chunk.
	 */
	public String type;

	/**
	 * Constructor.
	 * 
	 * @param tokenBeg
	 * @param tokenEnd
	 * @param type
	 */
	public TypedChunk(int tokenBeg, int tokenEnd, String type) {
		this.tokenBeg = tokenBeg;
		this.tokenEnd = tokenEnd;
		this.type = type;
	}

	@Override
	public boolean equals(Object obj) {
		if (!(obj instanceof TypedChunk))
			return false;
		TypedChunk other = (TypedChunk) obj;
		return tokenBeg == other.tokenBeg && tokenEnd == other.tokenEnd
				&& type.equals(other.type);
	}

	@Override
	public int hashCode() {
		Object[] array = { tokenBeg, tokenEnd, type };
		return HashCodeUtil.hash(HashCodeUtil.SEED, array);
	}

	@Override
	public int compareTo(TypedEntity e) {
		TypedChunk c = (TypedChunk) e;

		if (tokenBeg < c.tokenBeg)
			return -1;
		if (tokenBeg > c.tokenBeg)
			return 1;

		if (tokenEnd < c.tokenEnd)
			return -1;
		if (tokenEnd > c.tokenEnd)
			return 1;

		return type.compareTo(c.type);
	}

	@Override
	public String getType() {
		return type;
	}

}
