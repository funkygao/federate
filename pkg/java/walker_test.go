package java

import (
	"os"
	"path/filepath"
	"reflect"
	"sort"
	"testing"
)

// 创建临时目录结构的辅助函数
func createTempDir(t *testing.T) (string, func()) {
	tempDir, err := os.MkdirTemp("", "test")
	if err != nil {
		t.Fatalf("Failed to create temp dir: %v", err)
	}

	// 创建一些测试文件和目录
	files := []string{
		"file1.txt",
		"file2.xml",
		"dir1/file3.txt",
		"dir1/file4.xml",
		"dir2/file5.txt",
		"dir2/file6.xml",
	}

	for _, file := range files {
		path := filepath.Join(tempDir, file)
		err := os.MkdirAll(filepath.Dir(path), 0755)
		if err != nil {
			t.Fatalf("Failed to create directory: %v", err)
		}
		err = os.WriteFile(path, []byte("test content"), 0644)
		if err != nil {
			t.Fatalf("Failed to create file: %v", err)
		}
	}

	return tempDir, func() {
		os.RemoveAll(tempDir)
	}
}

func TestListFilesAsync(t *testing.T) {
	tempDir, cleanup := createTempDir(t)
	defer cleanup()

	tests := []struct {
		name      string
		predicate func(info os.FileInfo, path string) bool
		want      []string
	}{
		{
			name: "List all files",
			predicate: func(info os.FileInfo, path string) bool {
				return true
			},
			want: []string{
				"file1.txt",
				"file2.xml",
				filepath.Join("dir1", "file3.txt"),
				filepath.Join("dir1", "file4.xml"),
				filepath.Join("dir2", "file5.txt"),
				filepath.Join("dir2", "file6.xml"),
			},
		},
		{
			name:      "List only XML files",
			predicate: IsXML,
			want: []string{
				"file2.xml",
				filepath.Join("dir1", "file4.xml"),
				filepath.Join("dir2", "file6.xml"),
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			fileChan, errChan := ListFilesAsync(tempDir, tt.predicate)

			var got []string
			for file := range fileChan {
				relPath, err := filepath.Rel(tempDir, file.Path)
				if err != nil {
					t.Fatalf("Failed to get relative path: %v", err)
				}
				got = append(got, relPath)
			}

			if err := <-errChan; err != nil {
				t.Fatalf("ListFilesAsync returned an error: %v", err)
			}

			sort.Strings(got)
			sort.Strings(tt.want)

			if !reflect.DeepEqual(got, tt.want) {
				t.Errorf("ListFilesAsync() = %v, want %v", got, tt.want)
			}
		})
	}
}
