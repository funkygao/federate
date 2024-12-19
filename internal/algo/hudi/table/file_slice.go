package table

type FileSlice struct {
	BaseFile   string
	DeltaFiles []string
}

type FileGroup struct {
	PartitionPath string
	FileSlices    []FileSlice
}
