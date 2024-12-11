package io.github.federate.visitor;

import com.github.javaparser.ast.CompilationUnit;
import com.github.javaparser.ast.body.ClassOrInterfaceDeclaration;
import com.github.javaparser.ast.body.FieldDeclaration;
import com.github.javaparser.ast.body.MethodDeclaration;
import com.github.javaparser.ast.expr.*;
import com.github.javaparser.ast.visitor.Visitable;

import java.util.*;

public class TransformerResourceToAutowired extends BaseCodeModifier {
    private static final String RESOURCE = "Resource";
    private static final String AUTOWIRED = "Autowired";
    private static final String QUALIFIER = "Qualifier";

    private final Map<String, Map<String, Integer>> classBeanTypeCount = new HashMap<>();

    @Override
    public Visitable visit(CompilationUnit cu, Void arg) {
        classBeanTypeCount.clear();

        // 预处理：先统计所有字段的类型
        cu.findAll(FieldDeclaration.class).forEach(field -> {
            String beanType = field.getVariable(0).getTypeAsString();
            String className = field.findAncestor(ClassOrInterfaceDeclaration.class)
                    .map(c -> c.getNameAsString())
                    .orElse("");
            updateBeanTypeCount(beanType, className);
        });

        cu.addImport("org.springframework.beans.factory.annotation.Autowired");
        cu.addImport("org.springframework.beans.factory.annotation.Qualifier");

        return super.visit(cu, arg);
    }

    @Override
    public Visitable visit(FieldDeclaration field, Void arg) {
        // 创建注解的副本来避免并发修改异常
        List<AnnotationExpr> annotations = new ArrayList<>(field.getAnnotations());
        for (AnnotationExpr annotation : annotations) {
            if (isResourceAnnotation(annotation)) {
                transformResourceAnnotation(field, annotation);
                markDirty();
            }
        }
        return field;
    }

    @Override
    public Visitable visit(MethodDeclaration method, Void arg) {
        // 创建注解的副本来避免并发修改异常
        List<AnnotationExpr> annotations = new ArrayList<>(method.getAnnotations());
        for (AnnotationExpr annotation : annotations) {
            if (isResourceAnnotation(annotation)) {
                transformResourceAnnotation(method, annotation);
                markDirty();
            }
        }
        return method;
    }

    private void transformResourceAnnotation(FieldDeclaration field, AnnotationExpr annotation) {
        String beanType = field.getVariable(0).getTypeAsString();
        String className = field.findAncestor(ClassOrInterfaceDeclaration.class)
                .map(c -> c.getNameAsString())
                .orElse("");

        // 替换@Resource为@Autowired
        field.remove(annotation);
        field.addAnnotation(new MarkerAnnotationExpr(new Name(AUTOWIRED)));

        // 如果同类型的bean数量大于1，或有name属性，添加@Qualifier
        Integer typeCount = classBeanTypeCount.get(className).get(beanType);
        if (typeCount != null && typeCount > 1 || hasNameAttribute(annotation)) {
            String qualifierValue = getResourceName(annotation)
                    .orElse(field.getVariable(0).getNameAsString());
            field.addAnnotation(createQualifierAnnotation(qualifierValue));
        }
    }

    private void transformResourceAnnotation(MethodDeclaration method, AnnotationExpr annotation) {
        // 替换@Resource为@Autowired
        method.remove(annotation);
        method.addAnnotation(new MarkerAnnotationExpr(new Name(AUTOWIRED)));

        // 获取@Resource的name属性值或使用方法名生成的bean名称
        String qualifierValue = getResourceName(annotation)
                .orElseGet(() -> {
                    String methodName = method.getNameAsString();
                    if (methodName.startsWith("set")) {
                        // 去掉set前缀并将首字母小写
                        return methodName.substring(3, 4).toLowerCase() + methodName.substring(4);
                    }
                    return methodName;
                });

        method.addAnnotation(createQualifierAnnotation(qualifierValue));
    }

    private Optional<String> getResourceName(AnnotationExpr annotation) {
        if (annotation.isNormalAnnotationExpr()) {
            return ((NormalAnnotationExpr) annotation).getPairs().stream()
                    .filter(pair -> pair.getNameAsString().equals("name"))
                    .map(MemberValuePair::getValue)
                    .map(expr -> expr.asStringLiteralExpr().getValue())
                    .findFirst();
        } else if (annotation.isSingleMemberAnnotationExpr()) {
            return Optional.of(((SingleMemberAnnotationExpr) annotation)
                    .getMemberValue().asStringLiteralExpr().getValue());
        }
        return Optional.empty();
    }

    private void updateBeanTypeCount(String beanType, String className) {
        classBeanTypeCount
                .computeIfAbsent(className, k -> new HashMap<>())
                .merge(beanType, 1, Integer::sum);
    }

    private boolean hasNameAttribute(AnnotationExpr annotation) {
        if (annotation.isNormalAnnotationExpr()) {
            return ((NormalAnnotationExpr) annotation).getPairs().stream()
                    .anyMatch(pair -> pair.getNameAsString().equals("name"));
        } else if (annotation.isSingleMemberAnnotationExpr()) {
            return true; // 单值注解默认就是name属性
        }
        return false;
    }

    private AnnotationExpr createQualifierAnnotation(String value) {
        return new SingleMemberAnnotationExpr(
                new Name(QUALIFIER),
                new StringLiteralExpr(value)
        );
    }

    private boolean isResourceAnnotation(AnnotationExpr annotation) {
        return annotation.getNameAsString().equals(RESOURCE);
    }

}
