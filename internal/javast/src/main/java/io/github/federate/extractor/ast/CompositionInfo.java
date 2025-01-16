package io.github.federate.extractor.ast;

public class CompositionInfo {
    String containingClass;
    String composedClass;
    String fieldName;

    public CompositionInfo(String containingClass, String composedClass, String fieldName) {
        this.containingClass = containingClass;
        this.composedClass = composedClass;
        this.fieldName = fieldName;
    }
}
