package io.github.federate.extractor;

import com.github.javaparser.ast.visitor.VoidVisitorAdapter;
import com.google.gson.Gson;
import com.google.gson.GsonBuilder;

public abstract class BaseExtractor extends VoidVisitorAdapter<Void> {
    public abstract void export();

    protected final void export(Object data) {
        Gson gson = new GsonBuilder().create();
        System.out.println(gson.toJson(data));
    }
}
