package tagger.utils;

public class FileDescription {
public String encoding;
public String path;
public boolean compress;

public FileDescription(String path, String encoding, boolean compress) {
	this.path=path;
	this.encoding=encoding;
	this.compress=compress;
}
}
