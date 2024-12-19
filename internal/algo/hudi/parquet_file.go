package main

import (
	"github.com/xitongsys/parquet-go/source"
	"os"
)

type LocalParquetFile struct {
	file *os.File
}

func (f *LocalParquetFile) Create(name string) (source.ParquetFile, error) {
	file, err := os.Create(name)
	if err != nil {
		return nil, err
	}
	return &LocalParquetFile{file: file}, nil
}

func (f *LocalParquetFile) Open(name string) (source.ParquetFile, error) {
	file, err := os.Open(name)
	if err != nil {
		return nil, err
	}
	return &LocalParquetFile{file: file}, nil
}

func (f *LocalParquetFile) Seek(offset int64, whence int) (int64, error) {
	return f.file.Seek(offset, whence)
}

func (f *LocalParquetFile) Read(b []byte) (int, error) {
	return f.file.Read(b)
}

func (f *LocalParquetFile) Write(b []byte) (int, error) {
	return f.file.Write(b)
}

func (f *LocalParquetFile) Close() error {
	return f.file.Close()
}
