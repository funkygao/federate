JDK_HOME := /usr/local/Cellar/openjdk/23.0.1
JAVA_FILES = com/example/*.java
CPP := clang++
CPP_OPTS := -std=c++11 -O3
CPP_FILES := JniStreamEnhancer.cpp simd_avx.cpp simd_sse41.cpp simd_scalar.cpp

CPU_FEATURES := $(shell sysctl -a | grep machdep.cpu.features)
ifneq ($(findstring AVX2,$(CPU_FEATURES)),)
    CPP_OPTS += -mavx2 -D__AVX2__
endif
ifneq ($(findstring SSE4.1,$(CPU_FEATURES)),)
    CPP_OPTS += -msse4.1 -D__SSE4_1__
endif

build:
	$(CPP) -shared -fPIC -I"$(JDK_HOME)/include" -I"$(JDK_HOME)/include/darwin" $(CPP_OPTS) -o libjni_stream_enhancer.dylib $(CPP_FILES)
	javac $(JAVA_FILES)

run:build
	java -Djava.library.path=. com.example.Main

clean:
	rm -f libjni_stream_enhancer.dylib com/example/*.class
