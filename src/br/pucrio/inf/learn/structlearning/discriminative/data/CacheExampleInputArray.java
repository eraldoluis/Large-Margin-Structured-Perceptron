package br.pucrio.inf.learn.structlearning.discriminative.data;

import java.io.BufferedInputStream;
import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.RandomAccessFile;
import java.nio.ByteBuffer;
import java.nio.channels.Channels;
import java.nio.channels.FileChannel;
import java.util.ArrayList;
import java.util.Collection;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentLinkedDeque;
import java.util.concurrent.ConcurrentSkipListSet;

public class CacheExampleInputArray implements ExampleInputArray {	
	private static final int initialCapacity = 100;
	// private static final long MAX_SIZE = 0X000800000L;// 0X100000000L;// 8GB

	private FileChannel channel;
	private List<long[]> positionInFileOfExamples;
	private int numberExamples;
	private File file;
	private final long CACHE_SIZE;

	private Map<Integer, ExampleInput> cacheDPGSInput = new ConcurrentHashMap<Integer, ExampleInput>();
	private long numberMemoryLoad;
	private ConcurrentLinkedDeque<Integer> indexToLoad;
	private ConcurrentSkipListSet<Integer> indexLoaded;
	private Set<Integer> indexRemoved;
	private Thread producerThread;
	private Object notifyConsumer;
	private Object notifyProducer;
	

	private class DPGSInputRunnable implements Runnable {
		@Override
		public void run() {
			synchronized (notifyProducer) {
				while (true) {
					while (!indexToLoad.isEmpty()) {
						Integer i = indexToLoad.poll();

						if (!cacheDPGSInput.containsKey(i)) {
							long[] positionAndSize = getPositionAndSizeDPGSInput(i
									.intValue());
							ObjectInputStream ois = null;
							InputStream in = null;

							try {
								channel.position(positionAndSize[0]);
								in = Channels.newInputStream(channel);
								ois = new ObjectInputStream(
										new BufferedInputStream(in));

								/*
								 * System.out .println("Cache " +
								 * cacheDPGSInput.size() + " Tamanho do cache "
								 * + numberMemoryLoad +
								 * " Tamanho do novo cache " + (numberMemoryLoad
								 * + positionAndSize[1]));
								 */
								synchronized (indexLoaded) {

									while (numberMemoryLoad
											+ positionAndSize[1] > CACHE_SIZE) {

										Set<Entry<Integer, ExampleInput>> set = cacheDPGSInput
												.entrySet();

										Integer key;

										for (Iterator iterator = set.iterator(); iterator
												.hasNext();) {
											Entry<Integer, ExampleInput> entry = (Entry<Integer, ExampleInput>) iterator
													.next();

											key = entry.getKey();

											if (indexLoaded.contains(key)){
												continue;
											}

											long[] positionAndSize2 = positionInFileOfExamples
													.get(key.intValue());

											iterator.remove();
											numberMemoryLoad -= positionAndSize2[1];

											if (numberMemoryLoad
													+ positionAndSize[1] <= CACHE_SIZE) {
												break;
											}

										}

										if (numberMemoryLoad
												+ positionAndSize[1] > CACHE_SIZE) {
											try {
												indexLoaded.wait();
											} catch (InterruptedException e) {
											}
										}
									}
								}

								/*
								 * System.out .println("Cache " +
								 * cacheDPGSInput.size() + " Tamanho do cache "
								 * + numberMemoryLoad +
								 * " Tamanho do novo cache " + (numberMemoryLoad
								 * + positionAndSize[1]));
								 */

								numberMemoryLoad += positionAndSize[1];
								cacheDPGSInput.put(i,
										(ExampleInput) ois.readObject());
							} catch (IOException e) {
								// TODO Auto-generated catch block
								e.printStackTrace();
							} catch (ClassNotFoundException e) {
								// TODO Auto-generated catch block
								e.printStackTrace();
							} finally {
								/*
								 * if (in != null) try { in.close(); } catch
								 * (IOException e) { e.printStackTrace(); } if
								 * (ois != null) try { ois.close(); } catch
								 * (IOException e) { // TODO Auto-generated
								 * catch block e.printStackTrace(); }
								 */
							}
						}

						/*
						 * System.out.println("Memory Used: " +
						 * (numberMemoryLoad/(Math.pow(2, 30)))+" GB" +
						 * " Max size memory: " + (CACHE_SIZE/(Math.pow(2,
						 * 30)))+" GB");
						 */
						
						synchronized(indexRemoved){
							if(!indexRemoved.remove(i)){
								indexLoaded.add(i);
							}
						}

						synchronized (notifyConsumer) {
							notifyConsumer.notifyAll();
						}
					}
					try {
						notifyProducer.wait();
					} catch (InterruptedException e) {
					}
				}
			}
		}
	}

	public CacheExampleInputArray(long cacheSize, String fileName) throws IOException {
		positionInFileOfExamples = new ArrayList<long[]>(initialCapacity);
		numberExamples = 0;
		producerThread = new Thread(new DPGSInputRunnable());
		indexToLoad = new ConcurrentLinkedDeque<Integer>();
		indexLoaded = new ConcurrentSkipListSet<Integer>();
		indexRemoved = new HashSet<Integer>();
		notifyProducer = new Object();
		notifyConsumer = new Object();
		this.CACHE_SIZE = cacheSize;

		// CREATE FILE
		this.file = new File(fileName);

		RandomAccessFile rAccessFile = new RandomAccessFile(file, "rw");
		rAccessFile.setLength(0);
		// INSTANCE CHANNEL
		channel = rAccessFile.getChannel();

		producerThread.start();
	}

	private long[] getPositionAndSizeDPGSInput(int i) {
		return positionInFileOfExamples.get(i);
	}

	public ExampleInput get(int index) {
		ExampleInput input;
		boolean addToQueue = true;

		synchronized (notifyConsumer) {
			do {
				input = cacheDPGSInput.get(new Integer(index));

				if (input != null) {
					
					synchronized(indexRemoved){
						Integer indexObj = new Integer(index);
						if(!indexLoaded.remove(indexObj)){
							indexRemoved.add(indexObj);
						}
					}
					
					new Thread(new Runnable() {

						@Override
						public void run() {
							synchronized (indexLoaded) {
								indexLoaded.notify();
							}
						}
					}).start();

					return input;
				}
				

				/*
				 * // TODO - Add index to load when using get if (addToQueue) {
				 * load(new int[] { index }); addToQueue = false; }
				 */

				try {
					notifyConsumer.wait();
				} catch (InterruptedException e) {
				}
			} while (true);
		}

	}
	
	@Override
	public void put(ExampleInputArray input) throws IOException,
			DatasetException {
		input.loadInOrder();
		
		for (int i = 0; i < input.getNumberExamples(); i++) {
			put(input.get(i));
		}
	}

	public void put(ExampleInput input) throws IOException, DatasetException {

		ByteArrayOutputStream byteOutputStream = null;
		ObjectOutputStream objOutputStream = null;
		long position = channel.size();
		long size = 0;

		try {
			byteOutputStream = new ByteArrayOutputStream();
			objOutputStream = new ObjectOutputStream(byteOutputStream);
			objOutputStream.writeObject(input);
			objOutputStream.flush();

			byte[] postingBytes = byteOutputStream.toByteArray();

			size = postingBytes.length;

			if (size > CACHE_SIZE) {
				throw new DatasetException(
						"The sample size is larger than cache size."
								+ " Sample size : " + size + ", Cache size: "
								+ CACHE_SIZE);
			}

			ByteBuffer buffer = ByteBuffer.allocate(postingBytes.length);
			// prepare buffer to fill with data.
			buffer.clear();
			// write the bytes
			buffer.put(postingBytes);
			// prepare for writing
			buffer.flip();

			// Save input at end of file
			channel.position(position);
			channel.write(buffer);
		} finally {
			if (objOutputStream != null) {
				objOutputStream.close();
			}
			if (byteOutputStream != null) {
				byteOutputStream.close();
			}
		}

		positionInFileOfExamples.add(numberExamples, new long[] { position,
				size });
		numberExamples++;
	}

	public int getNumberExamples() {
		return numberExamples;
	}

	public void load(int[] index) {
		boolean insertNewIndex = false;

		for (int i : index) {
			Integer newIndex = new Integer(i);

			if (!indexToLoad.contains(newIndex)) {
				indexToLoad.offer(new Integer(i));
				insertNewIndex = true;
			}
		}
		
		if (insertNewIndex) {
			new Thread(new Runnable() {
				@Override
				public void run() {
					synchronized (notifyProducer) {
						notifyProducer.notify();
					}
				}
			}).start();
		}
	}
	
	public void loadInOrder() {
		int [] array = new int[getNumberExamples()];
		
		for (int i = 0; i < array.length; i++) {
			array[i] = i;
		}
		
		load(array);
	}

	@Override
	public void put(Collection<ExampleInput> inputs) throws IOException,
			DatasetException {
		for (ExampleInput exampleInput : inputs) {
			put(exampleInput);
		}
		
	}

	/*
	 * public void printLoad(){ System.out.println(indexToLoad); }
	 * 
	 * public void printLoaded(){ System.out.println(indexLoaded); }
	 * 
	 * public void printCacheKeys(){
	 * System.out.println(cacheDPGSInput.keySet().toString()); }
	 */
}
