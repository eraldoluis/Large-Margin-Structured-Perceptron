package tagger.features;

import java.util.Vector;

/**
 * An API for feature builders. Also store the feature encoding mapping
 * (int-string feature mapping).
 * 
 * TODO: solve the [] vector duality on checkConsistency and
 * 
 * TODO: @eraldo I think the feature builder functionality must not be together
 * the feature encoding (int-string mapping). In fact, the feature encoding
 * functionality is almost encapsulated within the Hmap class. I think this is
 * ok. What is not ok is to keep the Hmap inside the feature builder. I think is
 * more natural to keep this map inside the dataset class.
 * 
 * @author jordi
 * 
 */
public interface FeatureBuilder {

	public Vector<Vector<String>> extractFeatures(String[] W, String[] P,
			String[] L);

	public int[][] encode(Vector<Vector<String>> O_str, boolean secondorder);

	public String FSIS(int nid);

	public Integer FSIS_update_hmap(String sid);

	public Integer FSIS_update_hmap(String sid, boolean update);

	public int FSIS_add_update_hmap(String w);

	public int FSIS_size();

	public int FSIS_hsize(); // ?? h.size == V_STRING.size()

	public void dump();

	// public String LSIS(int nid);
	// public int LSIS_size();
	// public Vector<String> getListLabel();
	// public Vector<String> decode_tags(Vector<Integer> O_int_tags);
	// public String[] decode_tags(int[] O_int_tags);
	// public Vector<String> checkConsistency(Vector<Integer> IN);
	// public String[] checkConsistency(int[] IN);
	// public int LSIS_add_update_hmap(String elementAt);

}
