package io.github.federate.visitor;

import com.github.javaparser.ast.CompilationUnit;

import java.io.IOException;
import java.nio.file.Path;

interface FileVisitor {
    void visit(CompilationUnit cu, Path filePath) throws IOException;
}

