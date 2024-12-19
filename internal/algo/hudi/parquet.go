package main

import (
	"os"
	"path/filepath"

	"github.com/xitongsys/parquet-go/reader"
	"github.com/xitongsys/parquet-go/writer"
)

func (t *Table) WriteParquet(records []Record, filePath string) error {
	dir := filepath.Dir(filePath)
	if err := os.MkdirAll(dir, 0755); err != nil {
		return err
	}

	pf := &LocalParquetFile{}
	fw, err := pf.Create(filePath)
	if err != nil {
		return err
	}
	defer fw.Close()

	pw, err := writer.NewParquetWriter(fw, new(Record), 4)
	if err != nil {
		return err
	}

	for _, record := range records {
		if err := pw.Write(record); err != nil {
			return err
		}
	}

	if err := pw.WriteStop(); err != nil {
		return err
	}

	return nil
}

func (t *Table) ReadParquet(filePath string) ([]Record, error) {
	pf := &LocalParquetFile{}
	fr, err := pf.Open(filePath)
	if err != nil {
		return nil, err
	}
	defer fr.Close()

	pr, err := reader.NewParquetReader(fr, new(Record), 4)
	if err != nil {
		return nil, err
	}
	defer pr.ReadStop()

	num := int(pr.GetNumRows())
	records := make([]Record, num)

	if err := pr.Read(&records); err != nil {
		return nil, err
	}

	return records, nil
}
