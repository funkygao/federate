package merge

import (
	"os"
	"reflect"
	"testing"
)

func TestFindSystemGetPropertyKeys(t *testing.T) {
	testCases := []struct {
		name     string
		content  string
		expected []string
	}{
		{
			name: "Single property",
			content: `
                public class Test {
                    public static void main(String[] args) {
                        String prop = System.getProperty("java.home");
                    }
                }
            `,
			expected: []string{"java.home"},
		},
		{
			name: "Multiple properties",
			content: `
                public class Test {
                    public static void main(String[] args) {
                        String prop1 = System.getProperty("java.home");
                        String prop2 = System.getProperty("user.dir");
                        System.out.println(System.getProperty("os.name"));
                    }
                }
            `,
			expected: []string{"java.home", "user.dir", "os.name"},
		},
		{
			name: "No properties",
			content: `
                public class Test {
                    public static void main(String[] args) {
                        System.out.println("Hello, World!");
                    }
                }
            `,
			expected: []string{},
		},
		{
			name: "Mixed quotes",
			content: `
                public class Test {
                    public static void main(String[] args) {
                        String prop1 = System.getProperty("java.home");
                        String prop2 = System.getProperty('user.dir');
                    }
                }
            `,
			expected: []string{"java.home", "user.dir"},
		},
		{
			name: "Variable as argument",
			content: `
                public class Test {
                    public static void main(String[] args) {
                        String prop1 = System.getProperty("java.home");
                        String prop2 = System.getProperty(SystemConstants.APP_NAME);
                    }
                }
            `,
			expected: []string{"java.home", "VARIABLE: SystemConstants.APP_NAME"},
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			// 创建临时目录
			tempDir := t.TempDir()

			// 创建临时文件
			tmpfile, err := os.CreateTemp(tempDir, "test*.java")
			if err != nil {
				t.Fatalf("Cannot create temporary file: %v", err)
			}
			defer tmpfile.Close()

			// 写入测试内容
			if _, err := tmpfile.Write([]byte(tc.content)); err != nil {
				t.Fatalf("Failed to write to temporary file: %v", err)
			}

			// 运行测试函数
			got, err := findSystemGetPropertyKeys(tmpfile.Name())
			if err != nil {
				t.Fatalf("findSystemGetPropertyKeys returned an error: %v", err)
			}

			// 检查结果
			if !reflect.DeepEqual(got, tc.expected) {
				t.Errorf("findSystemGetPropertyKeys() = %v, want %v", got, tc.expected)
			}
		})
	}
}
