package io.github.federate.visitor;

import com.github.javaparser.ast.CompilationUnit;
import com.github.javaparser.ast.expr.AnnotationExpr;
import com.github.javaparser.ast.expr.SingleMemberAnnotationExpr;
import com.github.javaparser.ast.expr.StringLiteralExpr;
import com.github.javaparser.ast.visitor.ModifierVisitor;
import com.github.javaparser.ast.visitor.Visitable;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;

public class ValueAnnotationVisitor extends ModifierVisitor<Void> implements FileVisitor {
    private final String oldValue;
    private final String newValue;

    public ValueAnnotationVisitor(String oldValue, String newValue) {
        this.oldValue = oldValue;
        this.newValue = newValue;
    }

    @Override
    public Visitable visit(SingleMemberAnnotationExpr n, Void arg) {
        if (n.getNameAsString().equals("Value")) {
            if (n.getMemberValue() instanceof StringLiteralExpr) {
                StringLiteralExpr value = (StringLiteralExpr) n.getMemberValue();
                if (value.getValue().equals(oldValue)) {
                    return new SingleMemberAnnotationExpr(n.getName(), new StringLiteralExpr(newValue));
                }
            }
        }
        return super.visit(n, arg);
    }

    @Override
    public void visit(CompilationUnit cu, Path filePath) throws IOException {
        CompilationUnit modifiedCu = (CompilationUnit) cu.accept(this, null);
        Files.write(filePath, modifiedCu.toString().getBytes());
        System.out.println("Updated file: " + filePath);
    }

    @Override
    public ParserResult getResult(Path rootPath, Path filePath) {
        return null; // This visitor doesn't produce a ParserResult
    }
}

