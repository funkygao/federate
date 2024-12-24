#include <jni.h>
#include <smmintrin.h> // 包含 SSE4.1 指令集的头文件
#include <algorithm>   // 用于 std::max_element

extern "C" JNIEXPORT jlong JNICALL Java_JniStreamEnhancer_findMaxId(JNIEnv* env, jclass clazz, jlongArray ids) {
    jsize length = env->GetArrayLength(ids);

    // 获取数组元素
    jlong* array = env->GetLongArrayElements(ids, nullptr);

    // 使用 SSE2 指令集查找最大值
    __m128i max_vec = _mm_set1_epi64x(0); // 初始化最大值向量为 0
    for (jsize i = 0; i < length; i += 2) {
        __m128i vec = _mm_loadu_si128(reinterpret_cast<__m128i*>(&array[i]));

        // 提取高 32 位和低 32 位
        __m128i high_bits = _mm_srli_epi64(vec, 32);
        __m128i low_bits = _mm_and_si128(vec, _mm_set1_epi64x(0xFFFFFFFF));

        // 比较高 32 位
        __m128i max_high = _mm_max_epi32(high_bits, _mm_srli_epi64(max_vec, 32));

        // 比较低 32 位
        __m128i max_low = _mm_max_epi32(low_bits, _mm_and_si128(max_vec, _mm_set1_epi64x(0xFFFFFFFF)));

        // 合并结果
        max_vec = _mm_or_si128(_mm_slli_epi64(max_high, 32), max_low);
    }

    // 提取最大值
    alignas(16) long max_values[2];
    _mm_store_si128(reinterpret_cast<__m128i*>(max_values), max_vec);

    // 在 max_values 中找到最大值
    long max_id = *std::max_element(max_values, max_values + 2);

    // 释放数组元素
    env->ReleaseLongArrayElements(ids, array, JNI_ABORT);

    return max_id;
}

