package io.github.federate;

import com.github.javaparser.ast.body.ClassOrInterfaceDeclaration;
import com.github.javaparser.ast.body.MethodDeclaration;
import com.github.javaparser.ast.visitor.VoidVisitorAdapter;

import java.util.ArrayList;
import java.util.List;

public class JavaFileVisitor extends VoidVisitorAdapter<Void> {
    private List<String> classes = new ArrayList<>();
    private List<String> methods = new ArrayList<>();

    @Override
    public void visit(ClassOrInterfaceDeclaration n, Void arg) {
        classes.add(n.getNameAsString());
        super.visit(n, arg);
    }

    @Override
    public void visit(MethodDeclaration n, Void arg) {
        methods.add(n.getNameAsString());
        super.visit(n, arg);
    }

    public List<String> getClasses() {
        return classes;
    }

    public List<String> getMethods() {
        return methods;
    }
}

