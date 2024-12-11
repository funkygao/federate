package io.github.federate.visitor;

import com.github.javaparser.ast.CompilationUnit;
import com.github.javaparser.ast.body.ClassOrInterfaceDeclaration;
import com.github.javaparser.ast.visitor.ModifierVisitor;
import com.github.javaparser.ast.visitor.Visitable;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;

public abstract class BaseCodeModifier extends ModifierVisitor<Void> implements FileVisitor {
    protected String currentPackage;
    protected String currentClassName;

    private boolean modified = false;

    @Override
    public void visit(CompilationUnit cu, Path filePath) throws IOException {
        restart();
        CompilationUnit modifiedCu = (CompilationUnit) cu.accept(this, null);
        if (isModified()) {
            // Write file: CompilationUnit#toString will be the updated source code
            Files.write(filePath, modifiedCu.toString().getBytes());
            // System.out.println("Updated: " + filePath);
        }
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

    protected String getFQCN() {
        return currentPackage.isEmpty() ? currentClassName : currentPackage + "." + currentClassName;
    }

    protected void markDirty() {
        this.modified = true;
    }

    protected void restart() {
        this.modified = false;
    }

    public boolean isModified() {
        return modified;
    }
}

