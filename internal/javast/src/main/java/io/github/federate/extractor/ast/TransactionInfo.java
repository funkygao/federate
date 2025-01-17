package io.github.federate.extractor.ast;

public class TransactionInfo {
    String methodName;
    String fileName;
    int lineNumber;
    String type;

    public TransactionInfo(String methodName, String fileName, int lineNumber, String type) {
        this.methodName = methodName;
        this.fileName = fileName;
        this.lineNumber = lineNumber;
        this.type = type;
    }
}
