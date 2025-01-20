package io.github.federate.extractor.ast;

import java.util.List;

public class EnumInfo {
    String name;
    List<String> values;
    String fileName;

    public EnumInfo(String name, List<String> values, String fileName) {
        this.name = name;
        this.values = values;
        this.fileName = fileName;
    }
}
