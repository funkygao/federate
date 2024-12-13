package main

import (
	"time"
)

// Request represents an LLM inference request
type Request struct {
	ID           string
	Content      string
	PromptTokens int
	ArrivalTime  int64
	SLA          time.Duration
}
