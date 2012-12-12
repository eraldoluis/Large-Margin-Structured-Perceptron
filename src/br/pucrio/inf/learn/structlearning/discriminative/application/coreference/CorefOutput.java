package br.pucrio.inf.learn.structlearning.discriminative.application.coreference;

import br.pucrio.inf.learn.structlearning.discriminative.application.dp.data.DPOutput;
import br.pucrio.inf.learn.util.maxbranching.DisjointSets;

/**
 * Represent a coreference resolution output. That is a clustering of mentions
 * given in the input. Additionally, as a <code>DPOutput</code> subclass, also
 * represent a latent structure within each cluster. The latent structure is a
 * rooted tree (directed tree with a root node, i.e, there is a unique directed
 * path from the root node to every other node).
 * 
 * @author eraldo
 * 
 */
public class CorefOutput extends DPOutput {

	/**
	 * Auto-generated serial version ID.
	 */
	private static final long serialVersionUID = -6129033923429361340L;

	/**
	 * Represent the clustering. Each cluster is represented by some (arbitrary)
	 * index of a mention within this cluster.
	 */
	private DisjointSets clustering;

	/**
	 * Create an empty output with the given number of mentions.
	 * 
	 * @param numberOfMentions
	 */
	public CorefOutput(int numberOfMentions) {
		super(numberOfMentions);
		this.clustering = new DisjointSets(numberOfMentions);
	}

	@Override
	public DPOutput createNewObject() {
		return new CorefOutput(size());
	}

	/**
	 * Compute the clustering information from the underlying rooted tree.
	 * 
	 * @param root
	 *            the index of an artificial mention that is not within any
	 *            cluster. Thus, to include an edge from this mention to any
	 *            other (real) mention does not implicate cluster union. If this
	 *            value is less than zero, then it is ignored.
	 */
	public void computeClusteringFromTree(int root) {
		clustering.clear();
		for (int mentionRight = 0; mentionRight < size(); ++mentionRight) {
			int mentionLeft = getHead(mentionRight);
			if (mentionRight != root && mentionLeft != root && mentionLeft >= 0)
				clustering.union(mentionLeft, mentionRight);
		}
	}

	/**
	 * Return the id of the given cluster.
	 * 
	 * @param mention
	 * @return
	 */
	public int getClusterId(int mention) {
		return clustering.find(mention);
	}

	/**
	 * Conect two clusters that, thus, will become one cluster.
	 * 
	 * @param mentionLeft
	 * @param mentionRight
	 */
	public void connectClusters(int mentionLeft, int mentionRight) {
		clustering.union(mentionLeft, mentionRight);
	}

	/**
	 * Set the clustering of this object equal to the given objects's.
	 * 
	 * @param copy
	 */
	public void setClusteringEqualTo(CorefOutput copy) {
		this.clustering.setEqualTo(copy.clustering);
	}

}
