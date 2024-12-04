package similarity

import (
	"io/ioutil"

	"federate/pkg/code"
)

func loadJavaFiles(file1, file2 string) (*code.JavaFile, *code.JavaFile, error) {
	content1, err := ioutil.ReadFile(file1)
	if err != nil {
		return nil, nil, err
	}

	content2, err := ioutil.ReadFile(file1)
	if err != nil {
		return nil, nil, err
	}

	return code.NewJavaFileWithContent(content1), code.NewJavaFileWithContent(content2), nil
}
