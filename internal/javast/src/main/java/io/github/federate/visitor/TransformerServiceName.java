package io.github.federate.visitor;

import com.github.javaparser.ast.expr.*;
import com.github.javaparser.ast.visitor.Visitable;

import java.util.Arrays;
import java.util.List;
import java.util.Map;

// 修改 @Service, @Component 的 value
public class TransformerServiceName extends BaseCodeModifier {
    private final Map<String, String> serviceMap;
    private static final List<String> SUPPORTED_ANNOTATIONS = Arrays.asList("Service", "Component");

    public TransformerServiceName(Map<String, String> serviceMap) {
        this.serviceMap = serviceMap;
    }

    @Override
    public Visitable visit(SingleMemberAnnotationExpr n, Void arg) {
        return handleAnnotation(n);
    }

    @Override
    public Visitable visit(MarkerAnnotationExpr n, Void arg) {
        return handleAnnotation(n);
    }

    private Visitable handleAnnotation(AnnotationExpr n) {
        if (SUPPORTED_ANNOTATIONS.contains(n.getNameAsString()) && currentClassName != null) {
            String fqcn = getFQCN();
            String newValue = serviceMap.get(fqcn);
            if (newValue != null) {
                String oldValue = (n instanceof SingleMemberAnnotationExpr)
                    ? ((SingleMemberAnnotationExpr) n).getMemberValue().toString()
                    : "<empty>";
                modified = true;
                System.out.println("Modifying " + n.getNameAsString() + " for " + fqcn + ": " + oldValue + " -> " + newValue);
                return new SingleMemberAnnotationExpr(n.getName(), new StringLiteralExpr(newValue));
            }
        }
        return n;
    }
}

