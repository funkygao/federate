package io.github.federate.extractor;

import io.github.federate.extractor.ast.*;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class ASTInfo {
    List<String> imports = new ArrayList<>();
    List<String> classes = new ArrayList<>();
    List<String> methods = new ArrayList<>();
    List<String> staticMethodDeclarations = new ArrayList<>();
    List<String> methodCalls = new ArrayList<>();
    List<String> variables = new ArrayList<>();
    List<String> variableReferences = new ArrayList<>();

    Map<String, List<String>> inheritance = new HashMap<>();
    Map<String, List<String>> interfaces = new HashMap<>();
    List<String> annotations = new ArrayList<>();

    List<ComplexCondition> complexConditions = new ArrayList<>();
    List<CompositionInfo> compositions = new ArrayList<>();
    List<ComplexLoop> complexLoops = new ArrayList<>();
    List<FunctionalUsage> functionalUsages = new ArrayList<>();
    Map<String, FileStats> fileStats = new HashMap<>();
    List<LambdaInfo> lambdaInfos = new ArrayList<>();
    List<ReflectionUsage> reflectionUsages = new ArrayList<>();
    List<TransactionInfo> transactionInfos = new ArrayList<>();
    List<ExceptionCatchInfo> exceptionCatches = new ArrayList<>();
    List<MethodThrowsInfo> methodThrows = new ArrayList<>();
    List<EnumInfo> enums = new ArrayList<>();
    List<ConcurrencyUsage> concurrencyUsages = new ArrayList<>();
}
