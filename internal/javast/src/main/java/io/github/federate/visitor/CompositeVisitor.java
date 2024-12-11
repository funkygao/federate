package io.github.federate.visitor;

import com.github.javaparser.ast.CompilationUnit;
import com.github.javaparser.ast.NodeList;
import com.github.javaparser.ast.visitor.Visitable;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

public class CompositeVisitor extends BaseCodeModifier {
    private final List<BaseCodeModifier> visitors = new ArrayList<>();

    public void addVisitor(BaseCodeModifier visitor) {
        visitors.add(visitor);
    }

    @Override
    public void visit(CompilationUnit cu, Path filePath) throws IOException {
        restart();
        for (BaseCodeModifier visitor : visitors) {
            visitor.visit(cu, filePath);
            if (visitor.isModified()) {
                markDirty();
            }
        }

        if (isModified() && filePath != null) {
            Files.write(filePath, cu.toString().getBytes());
        }
    }
    
    @Override
    public Visitable visit(NodeList n, Void arg) {
        Visitable result = n;
        for (BaseCodeModifier visitor : visitors) {
            result = visitor.visit((NodeList<?>) result, arg);
        }
        return super.visit((NodeList<?>) result, arg);
    }
}
