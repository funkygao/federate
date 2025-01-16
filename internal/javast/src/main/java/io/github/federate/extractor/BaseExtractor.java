package io.github.federate.extractor;

import com.github.javaparser.ast.visitor.VoidVisitorAdapter;

public abstract class BaseExtractor extends VoidVisitorAdapter<Void> {
    public abstract void export();
}
