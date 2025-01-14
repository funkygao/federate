package io.github.federate.extractor;

public class LambdaInfo {
    int lineCount;
    int parameterCount;
    String context; // 方法名或类名
    String associatedStreamOp; // 相关的 Stream 操作
    String pattern; // 识别的模式
}
