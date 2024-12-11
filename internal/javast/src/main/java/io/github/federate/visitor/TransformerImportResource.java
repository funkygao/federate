package io.github.federate.visitor;

import com.github.javaparser.ast.NodeList;
import com.github.javaparser.ast.expr.*;
import com.github.javaparser.ast.visitor.Visitable;

import java.util.regex.Matcher;
import java.util.regex.Pattern;

public class TransformerImportResource extends BaseCodeModifier {
    private static final String ANNOTATION = "ImportResource";
    private static final String FEDERATED_DIR = "federated";

    private final String componentName;
    private final Pattern resourcePathPattern = Pattern.compile("(classpath:)?(/?)(.+)");

    public TransformerImportResource(String componentName) {
        this.componentName = componentName;
    }

    @Override
    public Visitable visit(SingleMemberAnnotationExpr n, Void arg) {
        if (ANNOTATION.equals(n.getNameAsString())) {
            Expression newValue = transformImportResource(n.getMemberValue());
            if (newValue != n.getMemberValue()) {
                n.setMemberValue(newValue);
                modified = true;
            }
        }
        return super.visit(n, arg);
    }

    @Override
    public Visitable visit(NormalAnnotationExpr n, Void arg) {
        if (ANNOTATION.equals(n.getNameAsString())) {
            for (MemberValuePair pair : n.getPairs()) {
                final String name = pair.getNameAsString();
                if ("locations".equals(name) || "value".equals(name)) {
                    Expression newValue = transformImportResource(pair.getValue());
                    if (newValue != pair.getValue()) {
                        pair.setValue(newValue);
                        modified = true;
                    }
                }
            }
        }
        return super.visit(n, arg);
    }

    private Expression transformImportResource(Expression expr) {
        if (expr instanceof StringLiteralExpr) {
            return transformSingleResource((StringLiteralExpr) expr);
        } else if (expr instanceof ArrayInitializerExpr) {
            ArrayInitializerExpr arrayExpr = (ArrayInitializerExpr) expr;
            NodeList<Expression> newValues = new NodeList<>();
            boolean changed = false;
            for (Expression e : arrayExpr.getValues()) {
                Expression newExpr = transformImportResource(e);
                if (newExpr != e) {
                    changed = true;
                }
                newValues.add(newExpr);
            }
            if (changed) {
                return new ArrayInitializerExpr(newValues);
            }
        }
        return expr;
    }

    private StringLiteralExpr transformSingleResource(StringLiteralExpr expr) {
        String value = expr.getValue();
        if (value.contains(FEDERATED_DIR + "/" + componentName + "/")) {
            return expr; // 如果路径已经包含了 federatedDir 和 componentName，不进行转换
        }

        Matcher matcher = resourcePathPattern.matcher(value);
        if (matcher.find()) {
            String prefix = matcher.group(1) != null ? matcher.group(1) : "";
            String leadingSlash = matcher.group(2);
            String resourcePath = matcher.group(3);
            String newResourcePath = String.format("%s%s%s/%s/%s", prefix, leadingSlash, FEDERATED_DIR, componentName, resourcePath);
            return new StringLiteralExpr(newResourcePath);
        }
        return expr;
    }
}
