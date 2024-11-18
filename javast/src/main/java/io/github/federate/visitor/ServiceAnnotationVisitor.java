package io.github.federate.visitor;

import com.github.javaparser.ast.CompilationUnit;
import com.github.javaparser.ast.body.ClassOrInterfaceDeclaration;
import com.github.javaparser.ast.expr.AnnotationExpr;
import com.github.javaparser.ast.expr.MarkerAnnotationExpr;
import com.github.javaparser.ast.expr.SingleMemberAnnotationExpr;
import com.github.javaparser.ast.expr.StringLiteralExpr;
import com.github.javaparser.ast.visitor.ModifierVisitor;
import com.github.javaparser.ast.visitor.Visitable;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Map;
import java.util.Optional;

public class ServiceAnnotationVisitor extends ModifierVisitor<Void> implements FileVisitor {
    private final Map<String, String> serviceMap;
    private boolean modified = false;
    private String currentPackage;
    private String currentClassName;

    public ServiceAnnotationVisitor(Map<String, String> serviceMap) {
        this.serviceMap = serviceMap;
    }

    @Override
    public Visitable visit(CompilationUnit cu, Void arg) {
        currentPackage = cu.getPackageDeclaration().map(pd -> pd.getName().asString()).orElse("");
        return super.visit(cu, arg);
    }

    @Override
    public Visitable visit(ClassOrInterfaceDeclaration n, Void arg) {
        currentClassName = n.getNameAsString();
        Visitable result = super.visit(n, arg);
        currentClassName = null; // Reset after visiting the class
        return result;
    }

    @Override
    public Visitable visit(SingleMemberAnnotationExpr n, Void arg) {
        if (n.getNameAsString().equals("Service") && currentClassName != null) {
            if (n.getMemberValue() instanceof StringLiteralExpr) {
                StringLiteralExpr value = (StringLiteralExpr) n.getMemberValue();
                String fqcn = getFQCN();
                String newValue = serviceMap.get(fqcn);
                if (newValue != null) {
                    modified = true;
                    return new SingleMemberAnnotationExpr(n.getName(), new StringLiteralExpr(newValue));
                }
            }
        }
        return super.visit(n, arg);
    }

    @Override
    public Visitable visit(MarkerAnnotationExpr n, Void arg) {
        if (n.getNameAsString().equals("Service") && currentClassName != null) {
            String fqcn = getFQCN();
            if (serviceMap.containsKey(fqcn)) {
                String newValue = serviceMap.get(fqcn);
                modified = true;
                return new SingleMemberAnnotationExpr(n.getName(), new StringLiteralExpr(newValue));
            }
        }
        return super.visit(n, arg);
    }

    private String getFQCN() {
        return currentPackage.isEmpty() ? currentClassName : currentPackage + "." + currentClassName;
    }

    @Override
    public void visit(CompilationUnit cu, Path filePath) throws IOException {
        modified = false; // Reset the flag for each file
        CompilationUnit modifiedCu = (CompilationUnit) cu.accept(this, null);
        if (modified) {
            Files.write(filePath, modifiedCu.toString().getBytes());
            System.out.println("Updated: " + filePath);
        }
    }

    @Override
    public ParserResult getResult(Path rootPath, Path filePath) {
        return null; // This visitor doesn't produce a ParserResult
    }
}

