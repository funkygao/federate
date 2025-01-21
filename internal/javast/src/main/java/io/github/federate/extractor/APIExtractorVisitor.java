package io.github.federate.extractor;

import com.github.javaparser.ast.CompilationUnit;
import com.github.javaparser.ast.body.ClassOrInterfaceDeclaration;
import com.github.javaparser.ast.body.MethodDeclaration;
import com.github.javaparser.ast.body.Parameter;
import io.github.federate.extractor.api.InterfaceInfo;
import io.github.federate.extractor.api.MethodInfo;
import io.github.federate.extractor.api.ParameterInfo;

import java.util.*;

public class APIExtractorVisitor extends BaseExtractor {
    private static final Set<String> ignoredMethodAnnotations = new HashSet<>();
    private static final Set<String> ignoredClassAnnotations = new HashSet<>();

    static {
        ignoredClassAnnotations.add("Deprecated");

        ignoredMethodAnnotations.add("Deprecated");
        ignoredMethodAnnotations.add("Debt");
        ignoredMethodAnnotations.add("Internal");
    }

    private final Map<String, InterfaceInfo> interfaces = new HashMap<>();

    @Override
    public void visit(CompilationUnit cu, Void arg) {
        super.visit(cu, arg);

        cu.findAll(ClassOrInterfaceDeclaration.class)
                .stream()
                .filter(ClassOrInterfaceDeclaration::isInterface)
                .filter(ClassOrInterfaceDeclaration::isPublic)
                .filter(this::hasNoIgnoredAnnotation)
                .forEach(this::processInterface);
    }

    private void processInterface(ClassOrInterfaceDeclaration n) {
        String interfaceName = n.getNameAsString();
        InterfaceInfo interfaceInfo = new InterfaceInfo();

        // 处理继承关系
        n.getExtendedTypes().forEach(extendedType ->
                interfaceInfo.extendedInterfaces.add(extendedType.getNameAsString()));

        n.getMethods().forEach(method -> {
            if (!hasIgnoredAnnotation(method)) {
                MethodInfo methodInfo = new MethodInfo();
                methodInfo.name = method.getNameAsString();
                methodInfo.parameters = getParameters(method);
                methodInfo.returnType = method.getType().asString();
                interfaceInfo.methods.add(methodInfo);
            }
        });

        interfaces.put(interfaceName, interfaceInfo);
    }

    private boolean hasNoIgnoredAnnotation(ClassOrInterfaceDeclaration n) {
        return n.getAnnotations().stream()
                .noneMatch(a -> ignoredClassAnnotations.contains(a.getNameAsString()));
    }

    private boolean hasIgnoredAnnotation(MethodDeclaration method) {
        return method.getAnnotations().stream()
                .anyMatch(a -> ignoredMethodAnnotations.contains(a.getNameAsString()));
    }

    private List<ParameterInfo> getParameters(MethodDeclaration method) {
        List<ParameterInfo> params = new ArrayList<>();
        for (Parameter param : method.getParameters()) {
            ParameterInfo paramInfo = new ParameterInfo();
            paramInfo.name = param.getNameAsString();
            paramInfo.type = param.getType().asString();
            params.add(paramInfo);
        }
        return params;
    }

    @Override
    public void export() {
        super.export(interfaces);
    }
}
