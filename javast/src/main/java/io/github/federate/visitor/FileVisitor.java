package io.github.federate.visitor;

import com.github.javaparser.ast.CompilationUnit;

import java.io.IOException;
import java.nio.file.Path;

public interface FileVisitor {
    void visit(CompilationUnit cu, Path filePath) throws IOException;

    ParserResult getResult(Path rootPath, Path filePath);
}

