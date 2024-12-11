// INPUT
import org.springframework.beans.factory.annotation.Value;

public class TestClass {
    @Value("${old.key:default} ${foo.bar:123}")
    private String property;
}

// EXPECTED
import org.springframework.beans.factory.annotation.Value;

public class TestClass {

    @Value("${new.key:default} ${bar.foo:123}")
    private String property;
}
