// INPUT
import javax.annotation.Resource;

public class TestClass {
    @Resource
    private TestService testService1;

    @Resource
    private TestService testService2;
}

// EXPECTED
import javax.annotation.Resource;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Qualifier;

public class TestClass {

    @Autowired
    @Qualifier("testService1")
    private TestService testService1;

    @Autowired
    @Qualifier("testService2")
    private TestService testService2;
}
