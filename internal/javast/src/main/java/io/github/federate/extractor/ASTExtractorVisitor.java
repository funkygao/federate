package io.github.federate.extractor;

import com.github.javaparser.ast.CompilationUnit;
import com.github.javaparser.ast.body.ClassOrInterfaceDeclaration;
import com.github.javaparser.ast.body.MethodDeclaration;
import com.github.javaparser.ast.body.VariableDeclarator;
import com.github.javaparser.ast.expr.MethodCallExpr;
import com.google.gson.Gson;
import com.google.gson.GsonBuilder;

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
    public void finish() {
        Gson gson = new GsonBuilder().create();
        System.out.println(gson.toJson(astInfo));
    }

    @Override
    public void visit(ClassOrInterfaceDeclaration n, Void arg) {
        astInfo.classes.add(n.getNameAsString());
        super.visit(n, arg);
    }

    @Override
    public void visit(MethodDeclaration n, Void arg) {
        astInfo.methods.add(n.getNameAsString());
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
}
