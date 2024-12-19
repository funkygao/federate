package table

import (
	"fmt"
	"path/filepath"
	"time"
)

type Compactor struct {
	table *Table
}

func NewCompactor(table *Table) *Compactor {
	return &Compactor{table: table}
}

func (c *Compactor) Compact() error {
	for partitionPath, fileGroup := range c.table.FileGroups {
		for i, slice := range fileGroup.FileSlices {
			if len(slice.DeltaFiles) >= 5 { // Threshold for compaction
				newBaseFile, err := c.mergeFileSlice(partitionPath, slice)
				if err != nil {
					return err
				}
				fileGroup.FileSlices[i] = FileSlice{BaseFile: newBaseFile}
			}
		}
	}

	instant := Instant{
		Timestamp: time.Now(),
		Action:    Compaction,
		State:     "COMPLETED",
	}
	c.table.Timeline.AddInstant(instant)
	return nil
}

func (c *Compactor) mergeFileSlice(partitionPath string, slice FileSlice) (string, error) {
	// In a real implementation, this would merge the base file and delta files
	// For this example, we'll just create a new base file name
	newBaseFileName := fmt.Sprintf("base_%d.parquet", time.Now().UnixNano())
	newBaseFilePath := filepath.Join(partitionPath, newBaseFileName)

	// Here you would:
	// 1. Read records from the base file
	// 2. Read and apply updates from delta files
	// 3. Write the merged records to the new base file
	// 4. Update the table's metadata and index

	// For now, we'll just pretend we did all that
	return newBaseFilePath, nil
}

func (c *Compactor) MergeSmallFiles(sizeThreshold int64) error {
	for partitionPath, fileGroup := range c.table.FileGroups {
		var smallFiles []string
		var totalSize int64

		for _, slice := range fileGroup.FileSlices {
			size, err := c.getFileSize(slice.BaseFile)
			if err != nil {
				return err
			}

			if size < sizeThreshold {
				smallFiles = append(smallFiles, slice.BaseFile)
				totalSize += size
			}

			if totalSize >= sizeThreshold || len(smallFiles) > 5 {
				newBaseFile, err := c.mergeFiles(partitionPath, smallFiles)
				if err != nil {
					return err
				}

				// Update file group with new base file
				fileGroup.FileSlices = append(fileGroup.FileSlices, FileSlice{BaseFile: newBaseFile})

				// Clear small files list
				smallFiles = nil
				totalSize = 0
			}
		}
	}

	instant := Instant{
		Timestamp: time.Now(),
		Action:    Compaction,
		State:     "COMPLETED",
	}
	c.table.Timeline.AddInstant(instant)

	return nil
}

func (c *Compactor) getFileSize(filePath string) (int64, error) {
	// In a real implementation, this would get the actual file size
	// For this example, we'll just return a dummy size
	return 1024 * 1024, nil // 1 MB
}

func (c *Compactor) mergeFiles(partitionPath string, filePaths []string) (string, error) {
	// In a real implementation, this would merge the given files
	// For this example, we'll just create a new base file name
	newBaseFileName := fmt.Sprintf("merged_%d.parquet", time.Now().UnixNano())
	newBaseFilePath := filepath.Join(partitionPath, newBaseFileName)

	// Here you would:
	// 1. Read records from all input files
	// 2. Merge the records, resolving any conflicts
	// 3. Write the merged records to the new base file
	// 4. Update the table's metadata and index

	// For now, we'll just pretend we did all that
	return newBaseFilePath, nil
}
