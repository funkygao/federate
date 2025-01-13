package io.github.federate.extractor;

import com.github.javaparser.ast.CompilationUnit;
import com.github.javaparser.ast.body.ClassOrInterfaceDeclaration;
import com.github.javaparser.ast.body.MethodDeclaration;
import com.github.javaparser.ast.body.VariableDeclarator;
import com.github.javaparser.ast.expr.AnnotationExpr;
import com.github.javaparser.ast.expr.MethodCallExpr;
import com.github.javaparser.ast.expr.NameExpr;
import com.google.gson.Gson;
import com.google.gson.GsonBuilder;

import java.util.ArrayList;

public class ASTExtractorVisitor extends BaseExtractor {
    private final ASTInfo astInfo;

    public ASTExtractorVisitor() {
        this.astInfo = new ASTInfo();
    }

    @Override
    public void visit(CompilationUnit cu, Void arg) {
        cu.getImports().forEach(i -> astInfo.imports.add(i.getNameAsString()));
        super.visit(cu, arg);
    }

    @Override
    public void visit(ClassOrInterfaceDeclaration n, Void arg) {
        String className = n.getNameAsString();
        astInfo.classes.add(className);

        // 处理继承
        n.getExtendedTypes().forEach(t -> astInfo.inheritance.computeIfAbsent(className, k -> new ArrayList<>()).add(t.getNameAsString()));

        // 处理接口实现
        n.getImplementedTypes().forEach(t -> astInfo.interfaces.computeIfAbsent(className, k -> new ArrayList<>()).add(t.getNameAsString()));

        // 处理注解
        n.getAnnotations().forEach(this::processAnnotation);

        super.visit(n, arg);
    }

    @Override
    public void visit(MethodDeclaration n, Void arg) {
        astInfo.methods.add(n.getNameAsString());
        n.getAnnotations().forEach(this::processAnnotation);
        super.visit(n, arg);
    }

    @Override
    public void visit(VariableDeclarator n, Void arg) {
        astInfo.variables.add(n.getNameAsString());
        super.visit(n, arg);
    }

    @Override
    public void visit(MethodCallExpr n, Void arg) {
        astInfo.methodCalls.add(n.getNameAsString());
        super.visit(n, arg);
    }

    private void processAnnotation(AnnotationExpr annotation) {
        String annotationName = annotation.getNameAsString();
        astInfo.annotations.add(annotationName);
    }

    @Override
    public void visit(NameExpr n, Void arg) {
        astInfo.variableReferences.add(n.getNameAsString());
        super.visit(n, arg);
    }

    @Override
    public void finish() {
        Gson gson = new GsonBuilder().create();
        System.out.println(gson.toJson(astInfo));
    }
}
