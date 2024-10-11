package merge

import (
	"io/ioutil"
	"os"
	"path/filepath"
	"testing"

	"federate/pkg/federated"
	"federate/pkg/manifest"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestXmlBeanManager_loadBeans(t *testing.T) {
	// 创建临时目录
	tempDir := t.TempDir()

	// 创建测试用的目录结构
	testComponentDir := filepath.Join(tempDir, federated.GeneratedResourceBaseDir("foo"), "testComponent")
	srcDir := testComponentDir
	err := os.MkdirAll(srcDir, 0755)
	require.NoError(t, err)

	// 创建测试用的 XML 文件
	testXml := `
<beans>
    <bean id="bean1" class="com.example.Bean1"/>
    <bean id="bean1" class="com.example.Foo"/>
    <util:list id="serialValidate.pick">
        <bean id="bean2" class="com.example.Bean2"/>
        <bean id="bean3" class="com.example.Bean3"/>
    </util:list>
    <util:list id="serialValidate.check">
        <bean id="bean2" class="com.example.Bean2"/>
        <bean id="bean3" class="com.example.Bean3"/>
    </util:list>
</beans>
`
	err = ioutil.WriteFile(filepath.Join(srcDir, "test.xml"), []byte(testXml), 0644)
	require.NoError(t, err)

	// 创建一个模拟的 Manifest
	m := &manifest.Manifest{
		Components: []manifest.ComponentInfo{
			{
				Name:    "testComponent",
				BaseDir: tempDir,
			},
		},
		Main: manifest.MainSystem{
			Name: "foo",
			Reconcile: manifest.ReconcileSpec{
				Resources: manifest.ResourcesReconcileSpec{
					FlatCopy: []string{},
				},
			},
		},
	}
	m.Components[0].M = &m.Main

	// 创建 XmlBeanManager
	beanManager := NewXmlBeanManager(m)
	err = beanManager.loadBeans()
	assert.NoError(t, err)

	// 验证结果
	assert.Len(t, beanManager.beanIdMap, 7, "Expected 7 items in beanIdMap, got %d", len(beanManager.beanIdMap))
	assert.Contains(t, beanManager.beanIdMap, "bean1", "beanIdMap should contain bean1")
	assert.Contains(t, beanManager.beanIdMap, "serialValidate.pick", "beanIdMap should contain serialValidate.pick")
	assert.Contains(t, beanManager.beanIdMap, "serialValidate.check", "beanIdMap should contain serialValidate.check")
	assert.Contains(t, beanManager.beanIdMap, "serialValidate.pick"+beanIdPathSeparator+"bean2", "beanIdMap should contain serialValidate.pick"+beanIdPathSeparator+"bean2")
	assert.Contains(t, beanManager.beanIdMap, "serialValidate.check"+beanIdPathSeparator+"bean2", "beanIdMap should contain serialValidate.check"+beanIdPathSeparator+"bean2")

	// 验证 bean1 的信息，是有重复的
	bean1s := beanManager.beanIdMap["bean1"]
	assert.Len(t, bean1s, 2)
	bean1Info := beanManager.beanIdMap["bean1"][0]
	assert.False(t, bean1Info.Nested())
	assert.Equal(t, "testComponent", bean1Info.ComponentName)
	assert.Equal(t, filepath.Join(testComponentDir, "test.xml"), bean1Info.SourceFilePath)
	assert.Equal(t, filepath.Join(m.Components[0].TargetResourceDir(), "test.xml"), bean1Info.TargetFilePath)
	assert.Equal(t, []string{"beans"}, bean1Info.ParentPath)

	// 验证 serialValidate.pick.bean2 的信息
	pickBean2Info := beanManager.beanIdMap["serialValidate.pick.bean2"][0]
	assert.Equal(t, "testComponent", pickBean2Info.ComponentName)
	assert.Equal(t, filepath.Join(testComponentDir, "test.xml"), pickBean2Info.SourceFilePath)
	assert.Equal(t, filepath.Join(m.Components[0].TargetResourceDir(), "test.xml"), pickBean2Info.TargetFilePath)
	assert.Equal(t, []string{"beans", "list"}, pickBean2Info.ParentPath)
	assert.True(t, pickBean2Info.Nested())

	// 验证 serialValidate.check.bean2 的信息
	checkBean2Info := beanManager.beanIdMap["serialValidate.check.bean2"][0]
	assert.Equal(t, "testComponent", checkBean2Info.ComponentName)
	assert.Equal(t, filepath.Join(testComponentDir, "test.xml"), checkBean2Info.SourceFilePath)
	assert.Equal(t, filepath.Join(m.Components[0].TargetResourceDir(), "test.xml"), checkBean2Info.TargetFilePath)
	assert.Equal(t, []string{"beans", "list"}, checkBean2Info.ParentPath)
	assert.True(t, checkBean2Info.Nested())
}
