package main

import (
	"log"
)

// LLMModel represents the interface for the actual LLM model
type LLMModel interface {
	PrefillForward(chunk []int) []byte
}

// LLMEngine handles the main LLM prefill process
type LLMEngine struct {
	model LLMModel
}

// PrefillResult represents the result of a prefill operation
type PrefillResult struct {
	RequestID       string
	KVCache         []byte
	PredictedLength LengthRange
}

// LengthRange represents the predicted length range for a request
type LengthRange struct {
	Min int
	Max int
}

func (e *LLMEngine) ChunkedPrefill(requests []Request, predictions map[string]LengthRange) []PrefillResult {
	var results []PrefillResult
	var currentChunk []int
	chunkMap := make(map[string]int) // Maps request IDs to their last prefilled token position

	log.Printf("Starting chunked prefill for %d requests", len(requests))

	for _, req := range requests {
		tokens := e.tokenize(req.Content)
		log.Printf("Processing request %s with %d tokens", req.ID, len(tokens))
		for i, token := range tokens {
			currentChunk = append(currentChunk, token)
			if len(currentChunk) == ChunkSize {
				kvCache := e.model.PrefillForward(currentChunk)
				e.updateResults(&results, chunkMap, req.ID, kvCache, predictions[req.ID])
				currentChunk = nil
			}
			chunkMap[req.ID] = i + 1
		}
	}

	// Process any remaining tokens in the last chunk
	if len(currentChunk) > 0 {
		// Pad the last chunk to ChunkSize
		for len(currentChunk) < ChunkSize {
			currentChunk = append(currentChunk, 0) // Padding with zeros
		}
		kvCache := e.model.PrefillForward(currentChunk)
		e.updateResults(&results, chunkMap, requests[len(requests)-1].ID, kvCache, predictions[requests[len(requests)-1].ID])
	}

	log.Printf("Completed chunked prefill, generated %d results", len(results))

	return results
}

func (e *LLMEngine) updateResults(results *[]PrefillResult, chunkMap map[string]int, requestID string, kvCache []byte, predictedLength LengthRange) {
	if existingResult := findResult(*results, requestID); existingResult != nil {
		// Append new KV cache to existing result
		existingResult.KVCache = append(existingResult.KVCache, kvCache...)
	} else {
		// Create new result
		*results = append(*results, PrefillResult{RequestID: requestID, KVCache: kvCache, PredictedLength: predictedLength})
	}
}

func findResult(results []PrefillResult, requestID string) *PrefillResult {
	for i := range results {
		if results[i].RequestID == requestID {
			return &results[i]
		}
	}
	return nil
}

func (e *LLMEngine) tokenize(content string) []int {
	return []int{}
}
