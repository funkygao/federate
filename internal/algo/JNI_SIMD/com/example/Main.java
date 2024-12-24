package com.example;

import java.util.Arrays;
import java.util.List;

public class Main {
    public static void main(String[] args) {
        List<Long> numbers = Arrays.asList(10L, 5L, 15L, 20L, 3L, 8L, 25L);

        long maxId = StreamEnhancer.enhanceStream(numbers.stream())
                .mapToLong(Long::longValue)
                .max()
                .orElse(0L);

        System.out.println("Max ID: " + maxId);
    }
}

