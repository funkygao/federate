package io.github.federate.visitor;

import com.github.javaparser.ast.CompilationUnit;
import com.github.javaparser.ast.body.ClassOrInterfaceDeclaration;
import com.github.javaparser.ast.body.MethodDeclaration;
import com.github.javaparser.ast.visitor.VoidVisitorAdapter;

import java.util.ArrayList;
import java.util.List;

import java.nio.file.Path;

public class ClassAndMethodAnalysisVisitor extends VoidVisitorAdapter<Void> implements FileVisitor {
    private List<String> classes = new ArrayList<>();
    private List<String> methods = new ArrayList<>();

    @Override
    public void visit(CompilationUnit cu, Path filePath) {
        super.visit(cu, null);
    }

    @Override
    public ParserResult getResult(Path rootPath, Path filePath) {
        return new ParserResult(rootPath.relativize(filePath).toString(), classes, methods);
    }

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

