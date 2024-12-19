package table

import (
	"github.com/bits-and-blooms/bloom/v3"
)

type Index interface {
	Add(key string, location FileLocation) error
	Get(key string) (FileLocation, bool)
	Remove(key string) error
}

type FileLocation struct {
	PartitionPath  string
	FileSliceIndex int
	IsBaseFile     bool
	FilePath       string
}

type BloomFilterIndex struct {
	filters   map[string]*bloom.BloomFilter
	locations map[string]FileLocation
}

func NewBloomFilterIndex() *BloomFilterIndex {
	return &BloomFilterIndex{
		filters:   make(map[string]*bloom.BloomFilter),
		locations: make(map[string]FileLocation),
	}
}

func (idx *BloomFilterIndex) Add(key string, location FileLocation) error {
	idx.locations[key] = location
	filter, exists := idx.filters[location.FilePath]
	if !exists {
		filter = bloom.NewWithEstimates(1000, 0.01)
		idx.filters[location.FilePath] = filter
	}
	filter.Add([]byte(key))
	return nil
}

func (idx *BloomFilterIndex) Get(key string) (FileLocation, bool) {
	location, exists := idx.locations[key]
	return location, exists
}

func (idx *BloomFilterIndex) Remove(key string) error {
	delete(idx.locations, key)
	return nil
}
