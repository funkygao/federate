package io.github.federate.extractor.info;

public class ReflectionUsage {
    String type;
    String name;
    String location;
    int lineNumber;

    public ReflectionUsage(String type, String name, String location, int lineNumber) {
        this.type = type;
        this.name = name;
        this.location = location;
        this.lineNumber = lineNumber;
    }
}
