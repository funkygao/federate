// INPUT
import javax.annotation.Resource;
import java.util.List;
import java.util.Map;

public class TestClass {
    @Resource
    private List<TestService> testServices;

    @Resource
    private Map<String, TestService> serviceMap;
}

// EXPECTED
import javax.annotation.Resource;
import java.util.List;
import java.util.Map;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Qualifier;

public class TestClass {

    @Autowired
    private List<TestService> testServices;

    @Autowired
    private Map<String, TestService> serviceMap;
}
