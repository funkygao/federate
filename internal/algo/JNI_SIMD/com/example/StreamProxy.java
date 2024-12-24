package com.example;

import java.lang.reflect.InvocationHandler;
import java.lang.reflect.Method;
import java.lang.reflect.Proxy;
import java.util.Optional;
import java.util.stream.Stream;

public class StreamProxy {
    @SuppressWarnings("unchecked")
    public static <T> Stream<T> create(Stream<T> stream) {
        return (Stream<T>) Proxy.newProxyInstance(
            Stream.class.getClassLoader(),
            new Class<?>[] { Stream.class },
            new StreamInvocationHandler<>(stream)
        );
    }

    private static class StreamInvocationHandler<T> implements InvocationHandler {
        private final Stream<T> stream;

        StreamInvocationHandler(Stream<T> stream) {
            this.stream = stream;
        }

        @Override
        public Object invoke(Object proxy, Method method, Object[] args) throws Throwable {
            if (method.getName().equals("max")) {
                Optional<Stream<T>> longStream = getStreamOfLong();
                if (longStream.isPresent()) {
                    return JniStreamEnhancer.findMaxId(
                        longStream.get().mapToLong(value -> ((Long) value).longValue()).toArray()
                    );
                }
            }
            return method.invoke(stream, args);
        }

        private Optional<Stream<T>> getStreamOfLong() {
            Optional<T> first = stream.findFirst();
            if (first.isPresent() && first.get() instanceof Long) {
                // 创建一个新的流，包含第一个元素和原始流的剩余部分
                return Optional.of(Stream.concat(Stream.of(first.get()), stream));
            }
            return Optional.empty();
        }
    }
}

