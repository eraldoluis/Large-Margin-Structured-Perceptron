package tagger.utils;

/**
 * @author jordi a quick fix to return a pair of objects
 * 
 * @param <TF>
 * @param <TS>
 */
public class Pair<TF, TS> {
	public TF first;;
	public TS second;

	public Pair(TF a, TS b) {
		first = a;
		second = b;
	}
}
