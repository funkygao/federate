import java.util.List;
import java.util.Arrays;
import java.util.stream.Collectors;
import java.util.stream.LongStream;

public class JniStreamEnhancer {

    static {
        System.loadLibrary("jni_stream_enhancer"); // 加载JNI库
    }

    // 声明JNI方法
    public static native long findMaxId(long[] ids);

    public static long getMaxId(List<Long> idList) {
        long[] ids = idList.stream().mapToLong(Long::longValue).toArray();
        return findMaxId(ids);
    }

    public static void main(String[] args) {
        List<Long> checkDetailList = Arrays.asList(10L, 30L, 20L, 40L, 50L);
        System.out.println("Max ID: " + getMaxId(checkDetailList));
    }
}

