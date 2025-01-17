package io.github.federate.extractor.ast;

import java.util.Set;
import java.util.TreeSet;

public class MethodThrowsInfo {
    private String className;
    private String methodName;
    private Set<String> thrownExceptions;

    public MethodThrowsInfo(String className, String methodName) {
        this.className = className;
        this.methodName = methodName;
        this.thrownExceptions = new TreeSet<>();
    }

    public void addException(String exceptionType) {
        thrownExceptions.add(exceptionType);
    }

    // Getters if needed
    public String getClassName() { return className; }
    public String getMethodName() { return methodName; }
    public Set<String> getThrownExceptions() { return thrownExceptions; }
}
