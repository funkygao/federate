package io.github.federate.extractor;

public class FileStats {
    String fileName;
    int netLinesOfCode;
    int methodCount;
    int fieldCount;

    public FileStats(String fileName, int netLinesOfCode, int methodCount, int fieldCount) {
        this.fileName = fileName;
        this.netLinesOfCode = netLinesOfCode;
        this.methodCount = methodCount;
        this.fieldCount = fieldCount;
    }
}
