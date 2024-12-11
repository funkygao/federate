// INPUT
import javax.annotation.Resource;
import lombok.Getter;

public class TestClass {
    @Getter
    @Resource
    private TestService testService;
}

// EXPECTED
import javax.annotation.Resource;
import lombok.Getter;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Qualifier;

public class TestClass {

    @Getter
    @Autowired
    private TestService testService;
}
