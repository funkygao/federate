// INPUT
import org.springframework.beans.factory.annotation.Value;

public class TestClass {
    @Value("${old.key:defaultValue} - ${foo.bar}")
    private String property;
}

// EXPECTED
import org.springframework.beans.factory.annotation.Value;

public class TestClass {

    @Value("${new.key:defaultValue} - ${bar.foo}")
    private String property;
}
