// INPUT
import javax.annotation.Resource;

public class TestClass {
    @Resource
    private TestService testService;
}

// EXPECTED
import javax.annotation.Resource;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Qualifier;

public class TestClass {

    @Autowired
    private TestService testService;
}
