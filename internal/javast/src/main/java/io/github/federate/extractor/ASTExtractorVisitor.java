package io.github.federate.extractor;

import com.github.javaparser.ast.CompilationUnit;
import com.github.javaparser.ast.Node;
import com.github.javaparser.ast.body.ClassOrInterfaceDeclaration;
import com.github.javaparser.ast.body.FieldDeclaration;
import com.github.javaparser.ast.body.MethodDeclaration;
import com.github.javaparser.ast.body.VariableDeclarator;
import com.github.javaparser.ast.expr.*;
import com.github.javaparser.ast.stmt.*;
import com.google.gson.Gson;
import com.google.gson.GsonBuilder;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.Set;
import java.util.concurrent.atomic.AtomicInteger;

public class ASTExtractorVisitor extends BaseExtractor {
    private static final int SWITCH_SIZE = 5;
    private static final int IF_SIZE = 3;
    private static final Set<String> STREAM_OPERATIONS;

    private int currentNestingLevel = 0;
    private final ASTInfo astInfo;

    static {
        STREAM_OPERATIONS = new HashSet<>(Arrays.asList(
                // 中间操作
                "filter", "map", "flatMap", "distinct", "sorted", "peek", "limit", "skip",
                "parallel", "sequential", "unordered",

                // 终端操作
                "forEach", "forEachOrdered", "toArray", "reduce", "collect", "min", "max",
                "count", "anyMatch", "allMatch", "noneMatch", "findFirst", "findAny",

                // Short-circuiting 操作
                "limit", "findFirst", "findAny", "anyMatch", "allMatch", "noneMatch",

                // 并行操作
                "parallelStream",

                // Optional 相关操作
                "ifPresent", "orElse", "orElseGet", "orElseThrow",

                // Collectors 类的静态方法
                "toList", "toSet", "toMap", "groupingBy", "joining", "counting",
                "summarizingInt", "summarizingLong", "summarizingDouble",

                // 其他常见的流操作或相关方法
                "stream", "of", "generate", "iterate",
                "mapToInt", "mapToLong", "mapToDouble",
                "flatMapToInt", "flatMapToLong", "flatMapToDouble",
                "boxed", "asLongStream", "asDoubleStream"
        ));
    }

    public ASTExtractorVisitor() {
        this.astInfo = new ASTInfo();
    }

    @Override
    public void visit(CompilationUnit cu, Void arg) {
        cu.getImports().forEach(i -> astInfo.imports.add(i.getNameAsString()));
        super.visit(cu, arg);

        String fileName = cu.getStorage().get().getFileName();
        int netLinesOfCode = countNetLinesOfCode(cu);
        int methodCount = cu.findAll(MethodDeclaration.class).size();
        int fieldCount = cu.findAll(FieldDeclaration.class).stream()
                .filter(field -> !field.isStatic())
                .mapToInt(field -> field.getVariables().size())
                .sum();

        astInfo.fileStats.put(fileName, new FileStats(fileName, netLinesOfCode, methodCount, fieldCount));
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
        if (isStreamOperation(n)) {
            analyzeFunctionalUsage(n, "stream", n.getNameAsString());
        }
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
    public void visit(ForStmt n, Void arg) {
        analyzeLoop(n, "for", n.getBody());
        super.visit(n, arg);
    }

    @Override
    public void visit(WhileStmt n, Void arg) {
        analyzeLoop(n, "while", n.getBody());
        super.visit(n, arg);
    }

    @Override
    public void visit(ForEachStmt n, Void arg) {
        analyzeLoop(n, "foreach", n.getBody());
        super.visit(n, arg);
    }

    private void analyzeLoop(Statement loopStmt, String loopType, Statement body) {
        currentNestingLevel++;

        MethodDeclaration method = loopStmt.findAncestor(MethodDeclaration.class).orElse(null);
        String methodName = method != null ? method.getNameAsString() : "Unknown";
        String fileName = loopStmt.findCompilationUnit().get().getStorage().get().getFileName();
        int lineNumber = loopStmt.getBegin().get().line;
        int bodySize = getBodySize(body);

        if (currentNestingLevel > 1 || bodySize > 10) { // 可以调整这些阈值
            astInfo.complexLoops.add(new ComplexLoop(methodName, fileName, loopType, lineNumber, currentNestingLevel, bodySize));
        }

        body.accept(this, null);
        currentNestingLevel--;
    }

    private int getBodySize(Statement body) {
        if (body instanceof BlockStmt) {
            return ((BlockStmt) body).getStatements().size();
        } else {
            return 1; // 如果循环体不是块语句，就当作一个语句计数
        }
    }

    @Override
    public void visit(LambdaExpr n, Void arg) {
        analyzeFunctionalUsage(n, "lambda", "lambda");

        LambdaInfo info = new LambdaInfo();
        info.lineCount = n.getEnd().get().line - n.getBegin().get().line + 1;
        info.parameterCount = n.getParameters().size();
        info.context = n.findAncestor(MethodDeclaration.class)
                .map(MethodDeclaration::getNameAsString)
                .orElse("Unknown");
        info.associatedStreamOp = findAssociatedStreamOp(n);
        info.pattern = identifyPattern(n);
        astInfo.lambdaInfos.add(info);
        
        super.visit(n, arg);
    }

    private boolean isStreamOperation(MethodCallExpr n) {
        String methodName = n.getNameAsString();

        // 检查方法名是否在我们的操作列表中
        if (STREAM_OPERATIONS.contains(methodName)) {
            return true;
        }

        // 检查是否是 Collectors 的静态方法调用
        if (n.getScope().isPresent() && n.getScope().get() instanceof NameExpr) {
            String scope = ((NameExpr) n.getScope().get()).getNameAsString();
            if ("Collectors".equals(scope) && STREAM_OPERATIONS.contains(methodName)) {
                return true;
            }
        }

        return false;
    }

    private void analyzeFunctionalUsage(Node n, String type, String operation) {
        MethodDeclaration method = n.findAncestor(MethodDeclaration.class).orElse(null);
        String methodName = method != null ? method.getNameAsString() : "Unknown";
        String fileName = n.findCompilationUnit().get().getStorage().get().getFileName();
        int lineNumber = n.getBegin().get().line;
        String context = getContext(n);

        astInfo.functionalUsages.add(new FunctionalUsage(methodName, fileName, lineNumber, type, operation, context));
    }

    private String getContext(Node n) {
        // 尝试获取上下文，例如方法调用的对象或变量名
        if (n.getParentNode().isPresent()) {
            Node parent = n.getParentNode().get();
            if (parent instanceof VariableDeclarator) {
                return ((VariableDeclarator) parent).getNameAsString();
            } else if (parent instanceof MethodCallExpr) {
                return ((MethodCallExpr) parent).getNameAsString();
            }
        }
        return "Unknown";
    }

    private int countNetLinesOfCode(CompilationUnit cu) {
        String[] lines = cu.toString().split("\n");
        int count = 0;
        boolean inBlockComment = false;
        for (String line : lines) {
            line = line.trim();
            if (line.startsWith("/*")) inBlockComment = true;
            if (!inBlockComment && !line.isEmpty() && !line.startsWith("//")) {
                count++;
            }
            if (line.endsWith("*/")) inBlockComment = false;
        }
        return count;
    }

    private String identifyPattern(LambdaExpr n) {
        if (n.getBody() instanceof ExpressionStmt) {
            Expression expr = ((ExpressionStmt) n.getBody()).getExpression();
            if (expr instanceof MethodCallExpr) {
                return "Method Call";
            } else if (expr instanceof BinaryExpr) {
                return "Condition";
            }
        } else if (n.getBody() instanceof BlockStmt) {
            BlockStmt block = (BlockStmt) n.getBody();
            if (block.getStatements().size() == 1 && block.getStatement(0) instanceof ReturnStmt) {
                return "Return";
            }
        } else if (n.getBody() instanceof ReturnStmt) {
            return "Return";
        }
        return "Other";
    }

    private String findAssociatedStreamOp(LambdaExpr n) {
        return n.getParentNode()
                .filter(parent -> parent instanceof MethodCallExpr)
                .map(parent -> ((MethodCallExpr) parent).getNameAsString())
                .orElse("Unknown");
    }

    @Override
    public void finish() {
        Gson gson = new GsonBuilder().create();
        System.out.println(gson.toJson(astInfo));
    }
}
