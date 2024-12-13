package main

// DecodeInstance represents a decode instance.
// It generates outputs token-by-token in an auto-regressive manner: decode phase, memory-bound.
type DecodeInstance struct {
	ID                 string
	AvailableResources int
	HeavyDecodeCount   int
	LightDecodeCount   int
}
