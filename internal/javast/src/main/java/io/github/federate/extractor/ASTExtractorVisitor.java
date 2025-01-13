package io.github.federate.extractor;

import com.github.javaparser.ast.CompilationUnit;
import com.github.javaparser.ast.body.ClassOrInterfaceDeclaration;
import com.github.javaparser.ast.body.FieldDeclaration;
import com.github.javaparser.ast.body.MethodDeclaration;
import com.github.javaparser.ast.body.VariableDeclarator;
import com.github.javaparser.ast.expr.*;
import com.github.javaparser.ast.stmt.IfStmt;
import com.github.javaparser.ast.stmt.SwitchStmt;
import com.google.gson.Gson;
import com.google.gson.GsonBuilder;

import java.util.ArrayList;
import java.util.concurrent.atomic.AtomicInteger;

public class ASTExtractorVisitor extends BaseExtractor {
    private static final int SWITCH_SIZE = 5;
    private static final int IF_SIZE = 3;

    private final ASTInfo astInfo;

    public ASTExtractorVisitor() {
        this.astInfo = new ASTInfo();
    }

    @Override
    public void visit(CompilationUnit cu, Void arg) {
        cu.getImports().forEach(i -> astInfo.imports.add(i.getNameAsString()));
        super.visit(cu, arg);
    }

    @Override
    public void visit(ClassOrInterfaceDeclaration n, Void arg) {
        String className = n.getNameAsString();
        astInfo.classes.add(className);

        // 处理继承
        n.getExtendedTypes().forEach(t -> astInfo.inheritance.computeIfAbsent(className, k -> new ArrayList<>()).add(t.getNameAsString()));

        // 处理接口实现
        n.getImplementedTypes().forEach(t -> astInfo.interfaces.computeIfAbsent(className, k -> new ArrayList<>()).add(t.getNameAsString()));

        // 处理注解
        n.getAnnotations().forEach(this::processAnnotation);

        super.visit(n, arg);
    }

    @Override
    public void visit(MethodDeclaration n, Void arg) {
        astInfo.methods.add(n.getNameAsString());
        n.getAnnotations().forEach(this::processAnnotation);
        super.visit(n, arg);
    }

    @Override
    public void visit(VariableDeclarator n, Void arg) {
        astInfo.variables.add(n.getNameAsString());
        super.visit(n, arg);
    }

    @Override
    public void visit(MethodCallExpr n, Void arg) {
        astInfo.methodCalls.add(n.getNameAsString());
        super.visit(n, arg);
    }

    private void processAnnotation(AnnotationExpr annotation) {
        String annotationName = annotation.getNameAsString();
        astInfo.annotations.add(annotationName);
    }

    @Override
    public void visit(NameExpr n, Void arg) {
        astInfo.variableReferences.add(n.getNameAsString());
        super.visit(n, arg);
    }

    @Override
    public void visit(IfStmt n, Void arg) {
        super.visit(n, arg);
        analyzeCondition(n.getCondition(), "if");
    }

    @Override
    public void visit(ConditionalExpr n, Void arg) {
        super.visit(n, arg);
        analyzeCondition(n.getCondition(), "ternary");
    }

    @Override
    public void visit(SwitchStmt n, Void arg) {
        super.visit(n, arg);
        analyzeSwitchStatement(n, "switch");
    }

    private void analyzeCondition(Expression condition, String type) {
        int complexity = calculateConditionComplexity(condition);
        if (complexity > IF_SIZE) {
            MethodDeclaration method = condition.findAncestor(MethodDeclaration.class).orElse(null);
            String methodName = method != null ? method.getNameAsString() : "Unknown";
            String fileName = condition.findCompilationUnit().get().getStorage().get().getFileName();
            astInfo.complexConditions.add(new ComplexCondition(
                    fileName,
                    methodName,
                    type,
                    complexity,
                    condition.getBegin().get().line
            ));
        }
    }

    private void analyzeSwitchStatement(SwitchStmt switchStmt, String type) {
        int complexity = switchStmt.getEntries().size();
        if (complexity > SWITCH_SIZE) {
            MethodDeclaration method = switchStmt.findAncestor(MethodDeclaration.class).orElse(null);
            String methodName = method != null ? method.getNameAsString() : "Unknown";
            String fileName = switchStmt.findCompilationUnit().get().getStorage().get().getFileName();
            astInfo.complexConditions.add(new ComplexCondition(
                    fileName,
                    methodName,
                    type,
                    complexity,
                    switchStmt.getBegin().get().line
            ));
        }
    }

    private int calculateConditionComplexity(Expression condition) {
        AtomicInteger complexity = new AtomicInteger(1);
        condition.walk(node -> {
            if (node instanceof BinaryExpr) {
                BinaryExpr.Operator op = ((BinaryExpr) node).getOperator();
                if (op == BinaryExpr.Operator.AND || op == BinaryExpr.Operator.OR) {
                    complexity.incrementAndGet();
                }
            }
        });
        return complexity.get();
    }

    @Override
    public void visit(FieldDeclaration field, Void arg) {
        super.visit(field, arg);

        ClassOrInterfaceDeclaration containingClass = field.findAncestor(ClassOrInterfaceDeclaration.class).orElse(null);
        if (containingClass != null) {
            String containingClassName = containingClass.getNameAsString();
            field.getVariables().forEach(variable -> {
                String composedClassName = variable.getType().asString();
                String fieldName = variable.getNameAsString();
                astInfo.compositions.add(new CompositionInfo(containingClassName, composedClassName, fieldName));
            });
        }
    }

    @Override
    public void finish() {
        Gson gson = new GsonBuilder().create();
        System.out.println(gson.toJson(astInfo));
    }
}
