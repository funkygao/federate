package io.github.federate.extractor;

import com.github.javaparser.StaticJavaParser;
import com.github.javaparser.ast.CompilationUnit;
import com.github.javaparser.ast.body.ClassOrInterfaceDeclaration;
import com.github.javaparser.ast.body.FieldDeclaration;
import com.github.javaparser.ast.body.MethodDeclaration;
import com.github.javaparser.ast.expr.MethodCallExpr;
import com.github.javaparser.ast.expr.NameExpr;
import com.github.javaparser.resolution.declarations.ResolvedMethodDeclaration;
import com.github.javaparser.symbolsolver.JavaSymbolSolver;
import com.github.javaparser.symbolsolver.resolution.typesolvers.CombinedTypeSolver;
import com.github.javaparser.symbolsolver.resolution.typesolvers.JavaParserTypeSolver;
import com.github.javaparser.symbolsolver.resolution.typesolvers.ReflectionTypeSolver;

import java.io.File;
import java.util.*;

public class ReferenceExtractor extends BaseExtractor {
    private final Map<String, Set<String>> referenceMap;
    private final String targetClassName;
    private final String targetMemberName;

    public ReferenceExtractor(String classNameOrWithField, String projectRoot) {
        this.referenceMap = new HashMap<>();

        CombinedTypeSolver typeSolver = createTypeSolver(projectRoot);
        JavaSymbolSolver symbolSolver = new JavaSymbolSolver(typeSolver);
        StaticJavaParser.getParserConfiguration().setSymbolResolver(symbolSolver);

        String[] parts = classNameOrWithField.split("#");
        this.targetClassName = parts[0];
        this.targetMemberName = parts.length > 1 ? parts[1] : null;
    }

    private CombinedTypeSolver createTypeSolver(String projectRoot) {
        CombinedTypeSolver combinedSolver = new CombinedTypeSolver();
        combinedSolver.add(new ReflectionTypeSolver());
        addSourceDirectories(combinedSolver, new File(projectRoot));
        return combinedSolver;
    }

    private void addSourceDirectories(CombinedTypeSolver combinedSolver, File directory) {
        if (directory.isDirectory()) {
            if (isSourceDirectory(directory)) {
                combinedSolver.add(new JavaParserTypeSolver(directory));
            }
            for (File subDir : directory.listFiles(File::isDirectory)) {
                addSourceDirectories(combinedSolver, subDir);
            }
        }
    }

    private boolean isSourceDirectory(File directory) {
        File[] javaFiles = directory.listFiles((dir, name) -> name.endsWith(".java"));
        return javaFiles != null && javaFiles.length > 0;
    }

    @Override
    public void visit(CompilationUnit cu, Void arg) {
        super.visit(cu, arg);

        cu.findAll(ClassOrInterfaceDeclaration.class).stream()
                .filter(cls -> cls.getNameAsString().equals(targetClassName) ||
                        cls.getFullyQualifiedName().map(fqn -> fqn.equals(targetClassName)).orElse(false))
                .forEach(cls -> {
                    if (targetMemberName == null) {
                        cls.getMethods().forEach(method -> collectMethodReferences(cu, method));
                        cls.getFields().forEach(field -> collectFieldReferences(cu, field));
                    } else {
                        cls.getMethodsByName(targetMemberName).forEach(method -> collectMethodReferences(cu, method));
                        cls.getFieldByName(targetMemberName).ifPresent(field -> collectFieldReferences(cu, field));
                    }
                });
    }

    private void collectMethodReferences(CompilationUnit cu, MethodDeclaration method) {
        try {
            String methodSignature = method.resolve().getQualifiedSignature();
            cu.findAll(MethodCallExpr.class).forEach(call -> {
                try {
                    ResolvedMethodDeclaration resolvedCall = call.resolve();
                    if (resolvedCall.getQualifiedSignature().equals(methodSignature)) {
                        String callerMethod = getCallerMethodSignature(call);
                        referenceMap.computeIfAbsent(methodSignature, k -> new HashSet<>()).add(callerMethod);
                    }
                } catch (Exception e) {
                    String unresolvedSignature = call.getNameAsString() + "()";
                    String callerMethod = getCallerMethodSignature(call);
                    referenceMap.computeIfAbsent(unresolvedSignature, k -> new HashSet<>()).add(callerMethod);
                }
            });
        } catch (Exception e) {
            // Ignore resolution errors
        }
    }

    private void collectFieldReferences(CompilationUnit cu, FieldDeclaration field) {
        String fieldName = field.getVariables().get(0).getNameAsString();
        String fieldSignature = targetClassName + "." + fieldName;
        cu.findAll(NameExpr.class).forEach(nameExpr -> {
            try {
                if (nameExpr.getNameAsString().equals(fieldName)) {
                    String callerMethod = getCallerMethodSignature(nameExpr);
                    referenceMap.computeIfAbsent(fieldSignature, k -> new HashSet<>()).add(callerMethod);
                }
            } catch (Exception e) {
                // Ignore resolution errors
            }
        });
    }

    private String getCallerMethodSignature(com.github.javaparser.ast.Node node) {
        return node.findAncestor(MethodDeclaration.class)
                .map(m -> {
                    try {
                        return m.resolve().getQualifiedSignature();
                    } catch (Exception e) {
                        return m.getNameAsString() + "()";
                    }
                })
                .orElse("Unknown");
    }

    @Override
    public void export() {
        super.export(referenceMap);
    }
}
