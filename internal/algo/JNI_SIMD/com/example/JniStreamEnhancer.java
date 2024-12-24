package com.example;

public class JniStreamEnhancer {
    static {
        System.loadLibrary("jni_stream_enhancer"); // 加载 JNI 库
    }

    // 声明 JNI 方法
    public static native long findMaxId(long[] ids);
}

