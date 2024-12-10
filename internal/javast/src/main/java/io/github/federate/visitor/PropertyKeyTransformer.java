package io.github.federate.visitor;

import com.github.javaparser.ast.NodeList;
import com.github.javaparser.ast.expr.*;
import com.github.javaparser.ast.visitor.Visitable;

import java.util.*;

/**
 * Transformer to update property keys in annotations.
 *
 * <p>根据 {@code keyMapping} 规则，oldKey -> newKey 改写Java源代码.</p>
 * <p>例如：`@Value("#{'${foo}'.split(',')}")`，keyMapping 里包含：foo -> egg.foo，则替换为：`@Value("#{'${egg.foo}'.split(',')}")`</p>
 */
public class PropertyKeyTransformer extends BaseCodeModifier {
    private final Set<String> annotationsToProcess = new HashSet<>(Arrays.asList(
            "Value",
            "ConfigurationProperties",
            "ConditionalOnProperty",
            "RequestMapping",
            "GetMapping",
            "PostMapping",
            "PutMapping",
            "DeleteMapping"
    ));
    private final Map<String, String> keyMapping;

    public PropertyKeyTransformer(Map<String, String> keyMapping) {
        this.keyMapping = keyMapping;
    }

    @Override
    public Visitable visit(SingleMemberAnnotationExpr n, Void arg) {
        transformAnnotation(n);
        return super.visit(n, arg);
    }

    @Override
    public Visitable visit(NormalAnnotationExpr n, Void arg) {
        transformAnnotation(n);
        return super.visit(n, arg);
    }

    private void transformAnnotation(AnnotationExpr n) {
        if (annotationsToProcess.contains(n.getNameAsString())) {
            if (n instanceof SingleMemberAnnotationExpr) {
                SingleMemberAnnotationExpr smae = (SingleMemberAnnotationExpr) n;
                Expression memberValue = smae.getMemberValue();
                Expression newMemberValue = transformExpression(memberValue);
                if (newMemberValue != memberValue) {
                    smae.setMemberValue(newMemberValue);
                    modified = true;
                }
            } else if (n instanceof NormalAnnotationExpr) {
                NormalAnnotationExpr nae = (NormalAnnotationExpr) n;
                boolean changed = false;
                for (MemberValuePair pair : nae.getPairs()) {
                    Expression oldValue = pair.getValue();
                    Expression newValue = transformExpression(oldValue);
                    if (newValue != oldValue) {
                        pair.setValue(newValue);
                        modified = true;
                        changed = true;
                    }
                }
                if (changed) {
                    modified = true;
                }
            }
        }
    }

    private Expression transformExpression(Expression expr) {
        if (expr instanceof StringLiteralExpr) {
            StringLiteralExpr sle = (StringLiteralExpr) expr;
            String newValue = replaceKeysInString(sle.getValue());
            if (!newValue.equals(sle.getValue())) {
                return new StringLiteralExpr(newValue);
            }
        } else if (expr instanceof ArrayInitializerExpr) {
            ArrayInitializerExpr aie = (ArrayInitializerExpr) expr;
            boolean changed = false;
            NodeList<Expression> values = new NodeList<>();
            for (Expression e : aie.getValues()) {
                Expression newExpr = transformExpression(e);
                if (newExpr != e) {
                    changed = true;
                }
                values.add(newExpr);
            }
            if (changed) {
                aie.setValues(values);
                return aie;
            }
        } else if (expr instanceof BinaryExpr) {
            BinaryExpr be = (BinaryExpr) expr;
            Expression left = transformExpression(be.getLeft());
            Expression right = transformExpression(be.getRight());
            if (left != be.getLeft() || right != be.getRight()) {
                be.setLeft(left);
                be.setRight(right);
                return be;
            }
        } else if (expr instanceof MethodCallExpr) {
            MethodCallExpr mce = (MethodCallExpr) expr;
            boolean changed = false;
            if (mce.getScope().isPresent()) {
                Expression newScope = transformExpression(mce.getScope().get());
                if (newScope != mce.getScope().get()) {
                    mce.setScope(newScope);
                    changed = true;
                }
            }
            NodeList<Expression> arguments = new NodeList<>();
            for (Expression arg : mce.getArguments()) {
                Expression newArg = transformExpression(arg);
                if (newArg != arg) {
                    changed = true;
                }
                arguments.add(newArg);
            }
            if (changed) {
                mce.setArguments(arguments);
                return mce;
            }
        } else if (expr instanceof NameExpr) {
            // Handle NameExpr if necessary
        } else if (expr instanceof FieldAccessExpr) {
            // Handle FieldAccessExpr if necessary
        } else if (expr instanceof ConditionalExpr) {
            // Handle ConditionalExpr if necessary
        } else if (expr instanceof EnclosedExpr) {
            EnclosedExpr ee = (EnclosedExpr) expr;
            Expression innerExpr = transformExpression(ee.getInner());
            if (innerExpr != ee.getInner()) {
                ee.setInner(innerExpr);
                return ee;
            }
        }
        // Handle other expression types if necessary
        return expr;
    }

    private String replaceKeysInString(String str) {
        String newStr = str;
        for (Map.Entry<String, String> entry : keyMapping.entrySet()) {
            String oldKey = entry.getKey();
            String newKey = entry.getValue();

            // Replace property placeholders
            newStr = newStr.replace("${" + oldKey + "}", "${" + newKey + "}");

            // Replace direct references (e.g., "key" -> "component.key")
            newStr = newStr.replace(oldKey, newKey);
        }
        return newStr;
    }
}

