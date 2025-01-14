package io.github.federate.extractor.info;

public class LambdaInfo {
    public int lineCount;
    public int parameterCount;
    public String context; // 方法名或类名
    public String associatedStreamOp; // 相关的 Stream 操作
    public String pattern; // 识别的模式
}
