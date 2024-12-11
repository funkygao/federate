// INPUT
import org.springframework.beans.factory.annotation.Value;

public class TestClass {
    @Value("${old.key}")
    private String prop1;

    @Value("${foo.bar}")
    private String prop2;

    @Value("${test.key}")
    private String prop3;
}

// EXPECTED
import org.springframework.beans.factory.annotation.Value;

public class TestClass {

    @Value("${new.key}")
    private String prop1;

    @Value("${bar.foo}")
    private String prop2;

    @Value("${key.test}")
    private String prop3;
}
