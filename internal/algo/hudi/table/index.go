package table

import (
	"encoding/json"
	"path/filepath"
)

type FSIndex struct {
	table *Table
}

func NewFSIndex(table *Table) *FSIndex {
	return &FSIndex{table: table}
}

func (idx *FSIndex) Add(key, filePath string) error {
	indexFilePath := filepath.Join(idx.table.Name, "index.json")

	var index map[string]string
	data, err := idx.table.FS.Read(indexFilePath)
	if err == nil {
		if err := json.Unmarshal(data, &index); err != nil {
			return err
		}
	} else {
		index = make(map[string]string)
	}

	index[key] = filePath

	updatedData, err := json.Marshal(index)
	if err != nil {
		return err
	}

	return idx.table.FS.Write(indexFilePath, updatedData)
}

func (idx *FSIndex) Get(key string) (string, bool) {
	indexFilePath := filepath.Join(idx.table.Name, "index.json")

	data, err := idx.table.FS.Read(indexFilePath)
	if err != nil {
		return "", false
	}

	var index map[string]string
	if err := json.Unmarshal(data, &index); err != nil {
		return "", false
	}

	filePath, ok := index[key]
	return filePath, ok
}

func (idx *FSIndex) Remove(key string) error {
	indexFilePath := filepath.Join(idx.table.Name, "index.json")

	data, err := idx.table.FS.Read(indexFilePath)
	if err != nil {
		return err
	}

	var index map[string]string
	if err := json.Unmarshal(data, &index); err != nil {
		return err
	}

	delete(index, key)

	updatedData, err := json.Marshal(index)
	if err != nil {
		return err
	}

	return idx.table.FS.Write(indexFilePath, updatedData)
}

func (idx *FSIndex) GetAll() (map[string]string, error) {
	indexFilePath := filepath.Join(idx.table.Name, "index.json")

	data, err := idx.table.FS.Read(indexFilePath)
	if err != nil {
		return nil, err
	}

	var index map[string]string
	if err := json.Unmarshal(data, &index); err != nil {
		return nil, err
	}

	return index, nil
}
