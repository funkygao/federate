package com.example;

import java.util.stream.Stream;

public class StreamEnhancer {
    public static <T> Stream<T> enhanceStream(Stream<T> stream) {
        return StreamProxy.create(stream);
    }
}

