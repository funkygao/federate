package property

import (
	"os"
	"path/filepath"
	"testing"

	"federate/pkg/manifest"
	"federate/pkg/merge/transformer"
	"federate/pkg/util"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"gopkg.in/yaml.v2"
)

func TestReplaceKeyInMatch(t *testing.T) {
	tests := []struct {
		name     string
		match    string
		key      string
		newKey   string
		expected string
	}{
		{
			name:     "Replace @Value annotation",
			match:    `@Value("${app.name}")`,
			key:      "app.name",
			newKey:   "component1.app.name",
			expected: `@Value("${component1.app.name}")`,
		},
		{
			name:     "Replace XML property",
			match:    `value="${db.url}"`,
			key:      "db.url",
			newKey:   "component2.db.url",
			expected: `value="${component2.db.url}"`,
		},
		{
			name:     "Replace @ConfigurationProperties annotation",
			match:    `@ConfigurationProperties("app")`,
			key:      "app",
			newKey:   "component3.app",
			expected: `@ConfigurationProperties("component3.app")`,
		},
	}

	task := &reconcileTask{}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := task.replaceKeyInMatch(tt.match, tt.key, tt.newKey)
			if result != tt.expected {
				t.Errorf("replaceKeyInMatch() = %v, want %v", result, tt.expected)
			}
		})
	}
}

func TestUpdateRequestMappingInFile(t *testing.T) {
	testCases := []struct {
		name        string
		input       string
		contextPath string
		expected    string
	}{
		{
			name:        "Simple RequestMapping",
			input:       `@RequestMapping("/api/users")`,
			contextPath: "/wms-stock",
			expected:    `@RequestMapping("/wms-stock/api/users")`,
		},
		{
			name:        "RequestMapping with value",
			input:       `@RequestMapping(value = "/api/users")`,
			contextPath: "/wms-stock",
			expected:    `@RequestMapping(value = "/wms-stock/api/users")`,
		},
		{
			name:        "Multiple RequestMappings",
			input:       `@RequestMapping("/api/users")\n@RequestMapping("/api/posts")`,
			contextPath: "/wms-stock",
			expected:    `@RequestMapping("/wms-stock/api/users")\n@RequestMapping("/wms-stock/api/posts")`,
		},
		{
			name:        "RequestMapping with existing context path",
			input:       `@RequestMapping("/wms-stock/api/users")`,
			contextPath: "/wms-stock",
			expected:    `@RequestMapping("/wms-stock/api/users")`,
		},
		{
			name:        "No RequestMapping",
			input:       `public class UserController {}`,
			contextPath: "/wms-stock",
			expected:    `public class UserController {}`,
		},
	}

	task := reconcileTask{}
	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			result := task.updateRequestMappingInFile(tc.input, tc.contextPath)
			if result != tc.expected {
				t.Errorf("Expected:\n%s\nGot:\n%s", tc.expected, result)
			}
		})
	}
}

func TestUpdateRequestMappingInFile_EdgeCases(t *testing.T) {
	testCases := []struct {
		name        string
		input       string
		contextPath string
		expected    string
	}{
		{
			name:        "Empty input",
			input:       "",
			contextPath: "/myapp",
			expected:    "",
		},
		{
			name:        "Empty context path",
			input:       `@RequestMapping("/api/users")`,
			contextPath: "",
			expected:    `@RequestMapping("/api/users")`,
		},
		{
			name:        "Context path without leading slash",
			input:       `@RequestMapping("/api/users")`,
			contextPath: "myapp",
			expected:    `@RequestMapping("myapp/api/users")`,
		},
		{
			name:        "RequestMapping with regex path",
			input:       `@RequestMapping("/api/{userId:[0-9]+}")`,
			contextPath: "/myapp",
			expected:    `@RequestMapping("/myapp/api/{userId:[0-9]+}")`,
		},
	}

	task := reconcileTask{}
	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			result := task.updateRequestMappingInFile(tc.input, tc.contextPath)
			assert.Equal(t, tc.expected, result, "The updated content should match the expected output")
		})
	}
}

func TestPropertyEntry(t *testing.T) {
	ps := PropertyEntry{FilePath: "foo.yaml"}
	assert.Equal(t, true, ps.IsYAML())
	assert.Equal(t, false, ps.IsProperties())
	ps.FilePath = "a/b/bar.properties"
	assert.Equal(t, true, ps.IsProperties())
	ps.FilePath = "a/b/egg.prOpeRties"
	assert.Equal(t, true, ps.IsProperties())
}

func TestPrepare(t *testing.T) {
	tempDir := t.TempDir()
	m := prepareTestManifest(t, tempDir)

	pm := NewManager(m)
	pm.silent = false
	pm.writeTarget = false
	require.NoError(t, pm.Prepare())
	conflicts := pm.identifyAllConflicts()
	if pm.debug {
		t.Logf("All properties:\n%s", util.Beautify(pm.r.resolvableEntries))
		t.Logf("Unresovled properties:\n%s", util.Beautify(pm.r.unresolvableEntries))
		t.Logf("Conflicts:\n%s", util.Beautify(conflicts))
	}

	// 验证无法解析的keys
	assert.Equal(t, "${non.exist}", pm.r.unresolvableEntries["b"]["wms.datasource.unresolved"].Value)
	assert.Equal(t, 1, len(pm.r.unresolvableEntries["b"]))

	// 验证冲突检测
	assert.Contains(t, conflicts, "datasource.mysql.url")
	assert.Contains(t, conflicts, "mysql.url")
	assert.Contains(t, conflicts, "mysql.maximumPoolSize")
	assert.Contains(t, conflicts, "wms.datasource.maximumPoolSize")
	assert.NotContains(t, conflicts, "a.key")
	assert.NotContains(t, conflicts, "b.key")
	assert.NotContains(t, conflicts, "mysql.driver")

	// 验证引用解析
	assert.Equal(t, 1234, pm.r.resolvableEntries["a"]["schedule.token"].Value) // 自己引用自己，只是properties引用yaml
	assert.Equal(t, "jdbc:mysql://1.1.1.1", pm.r.resolvableEntries["a"]["datasource.mysql.url"].Value)
	assert.Equal(t, "jdbc:mysql://1.1.1.1", pm.r.resolvableEntries["a"]["datasource.mysql.url"].Value)
	assert.Equal(t, "jdbc:mysql://1.1.1.8", pm.r.resolvableEntries["b"]["datasource.mysql.url"].Value)
	assert.Equal(t, "jdbc:mysql://1.1.1.9", pm.r.resolvableEntries["a"]["wms.reverse.datasource.ds0-master.jdbcUrl"].Value)

	assert.Equal(t, "jdbc:mysql://1.1.1.1", conflicts["datasource.mysql.url"]["a"])
	assert.Equal(t, "jdbc:mysql://1.1.1.8", conflicts["datasource.mysql.url"]["b"])

	// 不冲突
	assert.Equal(t, "foo", pm.r.resolvableEntries["a"]["a.key"].Value)
	assert.Equal(t, "0", pm.r.resolvableEntries["b"]["b.key"].Value)
	assert.Equal(t, "com.mysql.jdbc.Driver", pm.r.resolvableEntries["a"]["wms.datasource.driverClassName"].Value)

	// 调和冲突
	err := pm.Reconcile()
	require.NoError(t, err)
	if pm.debug {
		t.Logf("All properties after reconcile:\n%s", util.Beautify(pm.r.resolvableEntries))
	}

	// 检查解决后的属性
	resolvedProps := pm.r.resolvableEntries

	// 检查 datasource.mysql.url 是否被正确前缀化
	assert.Contains(t, resolvedProps["a"], "a.datasource.mysql.url")
	assert.Contains(t, resolvedProps["b"], "b.datasource.mysql.url")
	assert.Equal(t, "jdbc:mysql://1.1.1.1", resolvedProps["a"]["a.datasource.mysql.url"].Value)
	assert.Equal(t, "jdbc:mysql://1.1.1.8", resolvedProps["b"]["b.datasource.mysql.url"].Value)

	// 检查 mysql.maximumPoolSize 是否被正确前缀化
	assert.Contains(t, resolvedProps["a"], "a.mysql.maximumPoolSize")
	assert.Contains(t, resolvedProps["b"], "b.mysql.maximumPoolSize")
	// 该值仍引用
	assert.Equal(t, "${a.mysql.maximumPoolSize}", resolvedProps["a"]["a.wms.datasource.maximumPoolSize"].Raw)
	assert.Equal(t, "10", resolvedProps["a"]["a.mysql.maximumPoolSize"].Value)
	assert.Equal(t, "20", resolvedProps["b"]["b.mysql.maximumPoolSize"].Value)

	// 检查非冲突的键是否保持不变
	assert.Equal(t, "foo", resolvedProps["a"]["a.key"].Value)
	assert.Equal(t, "0", resolvedProps["b"]["b.key"].Value)

	// 检查原始的冲突键是否被删除
	assert.NotContains(t, resolvedProps["a"], "datasource.mysql.url")
	assert.Contains(t, resolvedProps["a"], "a.datasource.mysql.url")
	assert.NotContains(t, resolvedProps["b"], "datasource.mysql.url")
	assert.Contains(t, resolvedProps["b"], "b.datasource.mysql.url")

	// Resolve
	assert.Equal(t, "10", pm.Resolve("a.mysql.maximumPoolSize"))
	assert.Equal(t, "foo", pm.Resolve("a.key"))

	// ResolveLine
	assert.Equal(t, "hello 10", pm.ResolveLine("hello 10"))
	assert.Equal(t, "hello 10", pm.ResolveLine("hello ${a.mysql.maximumPoolSize}"))
	assert.Equal(t, "hello 10 and 20", pm.ResolveLine("hello ${a.mysql.maximumPoolSize} and ${b.mysql.maximumPoolSize}"))
}

func TestGenerateMergedYamlFile(t *testing.T) {
	tempDir := t.TempDir()
	m := prepareTestManifest(t, tempDir)

	pm := NewManager(m)
	pm.writeTarget = false
	require.NoError(t, pm.Prepare())

	mergedYamlPath := filepath.Join(tempDir, "merged.yml")
	pm.generateMergedYamlFile(mergedYamlPath)

	// 读取生成的YAML文件
	data, err := os.ReadFile(mergedYamlPath)
	require.NoError(t, err)
	t.Logf("%s", string(data))

	var mergedConfig map[string]interface{}
	err = yaml.Unmarshal(data, &mergedConfig)
	require.NoError(t, err)

	assert.Equal(t, "${mysql.maximumPoolSize}", mergedConfig["wms.datasource.maximumPoolSize"])

	// 检查不包含引用的值是否使用了解析后的实际值
	assert.Equal(t, "jdbc:mysql://1.1.1.1", mergedConfig["datasource.mysql.url"])
	assert.Equal(t, 1234, mergedConfig["schedule.token"])

	// 确保properties文件中的属性没有被包含
	assert.NotContains(t, mergedConfig, "mysql.url")
	assert.NotContains(t, mergedConfig, "a.key")
	assert.NotContains(t, mergedConfig, "b.key")
}

// properties引用了yml，yml也引用了properties
// 有冲突，也有不冲突
func prepareTestManifest(t *testing.T, tempDir string) *manifest.Manifest {
	a_properties := `
a.key=foo
mysql.driver=com.mysql.jdbc.Driver
mysql.url=${datasource.mysql.url}
mysql.maximumPoolSize=10
master.ds0.mysql.url=jdbc:mysql://a/db_1
reverse.master.ds0.mysql.url=${datasource.reverse.master.mysql.url}
schedule.token=${schedule.token}
`
	aRoot := filepath.Join(tempDir, "a")
	os.MkdirAll(aRoot, 0755)
	err := os.WriteFile(filepath.Join(aRoot, "application.properties"), []byte(a_properties), 0644)
	require.NoError(t, err)

	a_yaml := `
datasource:
  mysql:
    url: jdbc:mysql://1.1.1.1
  ds0-master:
    jdbcUrl: ${master.ds0.mysql.url}
  reverse:
    master:
      mysql:
        url: jdbc:mysql://1.1.1.9

schedule:
  token: 1234

wms:
  reverse:
    datasource:
      ds0-master:
        jdbcUrl: ${reverse.master.ds0.mysql.url}
  datasource:
    driverClassName: ${mysql.driver}
    maximumPoolSize: ${mysql.maximumPoolSize}
`
	err = os.WriteFile(filepath.Join(aRoot, "application.yml"), []byte(a_yaml), 0644)
	require.NoError(t, err)

	a_xml := `
<?xml version="1.0" encoding="UTF-8"?>
<beans xmlns="http://www.springframework.org/schema/beans"
       xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
       xsi:schemaLocation="http://www.springframework.org/schema/beans http://www.springframework.org/schema/beans/spring-beans.xsd">
    <bean id="checkPendingAsyncTask" class="com.google.wms.inventory.worker.AutoExecutionTask">
        <property name="name" value="检查是否有未执行的积压异步任务"/>
        <property name="foo" value="${wms.datasource.maximumPoolSize:5}"/>
        <property name="sqls" value="${wms.datasource.maximumPoolSize}"/>
    </bean>
    <kafka:transport address="${wms.datasource.maximumPoolSize}" password="${kafka.password}" app="${kafka.app}" user="${kafka.user}" sendTimeout="1000"/>
`
	os.WriteFile(filepath.Join(aRoot, "config.xml"), []byte(a_xml), 0644)

	b_properties := `
b.key=0
mysql.driver=com.mysql.jdbc.Driver
mysql.url=${datasource.mysql.url}
master.ds0.mysql.url=jdbc:mysql://b/db_2
mysql.maximumPoolSize=20
`
	bRoot := filepath.Join(tempDir, "b")
	os.MkdirAll(bRoot, 0755)
	err = os.WriteFile(filepath.Join(bRoot, "application.properties"), []byte(b_properties), 0644)
	require.NoError(t, err)

	b_yaml := `
datasource:
  mysql:
    url: jdbc:mysql://1.1.1.8
wms:
  ds0-master:
    jdbcUrl: ${master.ds0.mysql.url}
  ds1-master:
    jdbcUrl: ${master.ds0.mysql.url}
  datasource:
    driverClassName: ${mysql.driver}
    maximumPoolSize: ${mysql.maximumPoolSize}
    unresolved: ${non.exist}
`
	err = os.WriteFile(filepath.Join(bRoot, "application.yml"), []byte(b_yaml), 0644)
	require.NoError(t, err)

	b_xml := `
<?xml version="1.0" encoding="UTF-8"?>
<beans xmlns="http://www.springframework.org/schema/beans"
       xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
       xsi:schemaLocation="http://www.springframework.org/schema/beans http://www.springframework.org/schema/beans/spring-beans.xsd">
    <bean id="checkPendingAsyncTask" class="com.google.wms.stock.worker.AutoExecutionTask">
        <property name="name" value="检查是否有未执行的积压异步任务"/>
        <property name="foo" value="${wms.datasource.maximumPoolSize:5}"/>
        <property name="sqls" value="${wms.datasource.maximumPoolSize}"/>
    </bean>
    <kafka:transport address="${wms.datasource.maximumPoolSize}" password="${kafka.password}" app="${kafka.app}" user="${kafka.user}" sendTimeout="1000"/>
`
	os.WriteFile(filepath.Join(bRoot, "config.xml"), []byte(b_xml), 0644)

	m := &manifest.Manifest{}
	target := &manifest.MainSystem{}
	m.Components = []manifest.ComponentInfo{
		{
			M:       target,
			Name:    "a",
			BaseDir: tempDir,
			Resources: manifest.ComponentResourceSpec{
				BaseDirs: []string{""},
				PropertySources: []string{
					"application.properties",
				},
			},
		},
		{
			M:       target,
			Name:    "b",
			BaseDir: tempDir,
			Resources: manifest.ComponentResourceSpec{
				BaseDirs: []string{""},
				PropertySources: []string{
					"application.properties",
				},
			},
		},
	}
	return m
}

func showConflict(t *testing.T, conflicts map[string]map[string]interface{}) {
	for key, componentValue := range conflicts {
		for _, c := range util.MapSortedStringKeys(componentValue) {
			t.Logf("[%s] %s = %v", c, key, conflicts[key][c])
		}
	}
}

func TestUpdatedReferenceString(t *testing.T) {
	componentName, oldKey, newKey := "wms-stock", "db.pool.size", "wms-stock.db.pool.size"
	entry := PropertyEntry{Raw: "${db.pool.size}"}
	r := newRegistry(nil, nil, true)
	assert.Equal(t, "${wms-stock.db.pool.size}", r.updatedReferenceString(componentName, oldKey, newKey, entry))
}

func TestGenerateAllForManualCheck(t *testing.T) {
	if !testing.Verbose() {
		return
	}

	m := prepareTestManifest(t, "target")
	pm := NewManager(m)
	pm.writeTarget = false
	pm.Debug()
	pm.Prepare()

	t.Log("")
	t.Logf("属性冲突")
	// master.ds0.mysql.url
	// mysql.url
	showConflict(t, pm.IdentifyPropertiesFileConflicts())
	t.Logf("YML冲突")
	// wms.datasource.maximumPoolSize
	showConflict(t, pm.IdentifyYamlFileConflicts())
	t.Log("")

	t.Logf("Reconcile")
	pm.Reconcile()
	//t.Logf("%s", util.Beautify(pm.r.resolvableEntries))
	// a 和 b 的 yaml key: wms.datasource.maximumPoolSize，都在引用 ${mysql.maximumPoolSize}
	assert.Equal(t, "${a.mysql.maximumPoolSize}", pm.r.resolvableEntries["a"]["a.wms.datasource.maximumPoolSize"].Raw)
	assert.Equal(t, "10", pm.r.resolvableEntries["a"]["a.wms.datasource.maximumPoolSize"].Value)
	assert.Equal(t, "20", pm.r.resolvableEntries["b"]["b.wms.datasource.maximumPoolSize"].Value)
	assert.Equal(t, "${b.master.ds0.mysql.url}", pm.r.resolvableEntries["b"]["wms.ds0-master.jdbcUrl"].Raw)
	assert.Equal(t, "jdbc:mysql://b/db_2", pm.r.resolvableEntries["b"]["wms.ds1-master.jdbcUrl"].Value)
	t.Log("")

	t.Logf("Registry DUMP")
	pm.r.dump()
	t.Log("")

	pm.generateMergedYamlFile("target/application.yml")
	pm.generateMergedPropertiesFile("target/application.properties")
	t.Log("")

	t.Log("Summary")
	transformer.Get().ShowSummary()
}
