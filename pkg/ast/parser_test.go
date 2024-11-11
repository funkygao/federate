package ast

import (
	"testing"

	"federate/pkg/ast/parser"
	"github.com/stretchr/testify/assert"
)

// mockListener 是一个用于测试的模拟监听器
type mockListener struct {
	parser.BaseJava8ParserListener
	classCount  int
	methodCount int
}

func (l *mockListener) EnterClassDeclaration(ctx *parser.ClassDeclarationContext) {
	l.classCount++
}

func (l *mockListener) EnterMethodDeclaration(ctx *parser.MethodDeclarationContext) {
	l.methodCount++
}

func TestParser_ParseJava(t *testing.T) {
	tests := []struct {
		name            string
		javaSrc         string
		expectedClasses int
		expectedMethods int
		expectError     bool
	}{
		{
			name: "Valid Java code - single class with one method",
			javaSrc: `
				public class Test {
					public void method() {
						System.out.println("Hello, World!");
					}
				}
			`,
			expectedClasses: 1,
			expectedMethods: 1,
			expectError:     false,
		},
		{
			name: "Valid Java code - two classes with multiple methods",
			javaSrc: `
				public class Test1 {
					public void method1() {}
					private int method2() { return 0; }
				}
				class Test2 {
					protected String method3(int arg) { return ""; }
				}
			`,
			expectedClasses: 2,
			expectedMethods: 3,
			expectError:     false,
		},
		{
			name:            "Empty input",
			javaSrc:         "",
			expectedClasses: 0,
			expectedMethods: 0,
			expectError:     false,
		},
		{
			name: "Syntax error - missing semicolon",
			javaSrc: `
				public class Test {
					public void method() {
						System.out.println("Missing semicolon")
					}
				}
			`,
			expectedClasses: 0,
			expectedMethods: 0,
			expectError:     true,
		},
		{
			name: "Syntax error - unmatched brace",
			javaSrc: `
				public class Test {
					public void method() {
						System.out.println("Unmatched brace");
					}
				
			`,
			expectedClasses: 0,
			expectedMethods: 0,
			expectError:     true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			parser := NewParser()
			listener := &mockListener{}

			err := parser.ParseJava(tt.javaSrc, listener)

			if tt.expectError {
				assert.Error(t, err, "Expected an error, but got none")
			} else {
				assert.NoError(t, err, "Unexpected error")
				assert.Equal(t, tt.expectedClasses, listener.classCount, "Incorrect number of classes")
				assert.Equal(t, tt.expectedMethods, listener.methodCount, "Incorrect number of methods")
			}
		})
	}
}

func TestNewParser(t *testing.T) {
	parser := NewParser()
	assert.NotNil(t, parser, "NewParser() returned nil")
}

func TestParser_ErrorHandling(t *testing.T) {
	parser := NewParser()
	listener := &mockListener{}

	javaSrc := `
		public class Test {
			public void method() {
				int x = 10
				System.out.println(x);
			}
		}
	`

	err := parser.ParseJava(javaSrc, listener)
	assert.Error(t, err, "Expected an error due to missing semicolon, but got none")
	assert.Contains(t, err.Error(), "missing ';'", "Error message does not contain expected content")
}
