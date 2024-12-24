JDK_HOME := /usr/local/Cellar/openjdk/23.0.1
CPP := clang++
CPP_OPTS := -std=c++11

CPU_FEATURES := $(shell sysctl -a | grep machdep.cpu.features | grep AVX)
ifneq ($(CPU_FEATURES),)
    AVX_OPT := -mavx    # 使用 AVX 指令集
    SSE_OPT := -msse4.1 # 使用 SSE4.1 指令集
    CPP_OPTS += $(SSE_OPT) $(AVX_OPT)
else
    SSE_OPT := -msse4.1
    CPP_OPTS += $(SSE_OPT)
endif

build:
	$(CPP) -shared -fPIC -I"$(JDK_HOME)/include" -I"$(JDK_HOME)/include/darwin" $(CPP_OPTS) -o libjni_stream_enhancer.dylib JniStreamEnhancer.cpp
	javac JniStreamEnhancer.java

run:build
	java -Djava.library.path=. JniStreamEnhancer

clean:
	rm -f libjni_stream_enhancer.dylib JniStreamEnhancer.class
