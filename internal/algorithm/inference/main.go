package main

import (
	"fmt"
	"log"
	"sync"
	"time"
)

const (
	ChunkSize        = 512
	DecodeInstances  = 5
	PrefillInstances = 4
)

func main() {
	log.Println("Starting LLM inference system")

	clusterMonitor := NewClusterMonitor()

	for i := 0; i < DecodeInstances; i++ {
		decodeInstance := &DecodeInstance{
			ID:                 fmt.Sprintf("decode-%d", i),
			AvailableResources: 1000,
		}
		clusterMonitor.AddDecodeInstance(decodeInstance)
		log.Printf("Added decode instance: %s", decodeInstance.ID)
	}

	globalScheduler := NewGlobalScheduler(PrefillInstances, clusterMonitor)
	log.Printf("Created global scheduler with %d prefill instances", len(globalScheduler.PrefillInstances))

	// Add some sample requests
	requests := []Request{
		{ID: "1", Content: "Short request", PromptTokens: 10, ArrivalTime: time.Now().UnixNano(), SLA: 5 * time.Second},
		{ID: "2", Content: "Medium request", PromptTokens: 50, ArrivalTime: time.Now().UnixNano(), SLA: 3 * time.Second},
		{ID: "3", Content: "Long request", PromptTokens: 100, ArrivalTime: time.Now().UnixNano(), SLA: 4 * time.Second},
	}

	for _, request := range requests {
		globalScheduler.HandleIncomingRequest(request)
		log.Printf("Global scheduler handled incoming Request[%s] with %d prompt tokens", request.ID, request.PromptTokens)
	}

	// Process all requests
	var wg sync.WaitGroup
	for i, instance := range globalScheduler.PrefillInstances {
		wg.Add(1)
		go func(idx int, pi *PrefillInstance) {
			defer wg.Done()

			log.Printf("Starting processing for prefill instance %d", idx)
			t0 := time.Now()
			pi.ProcessRequests()
			log.Printf("Prefill instance %d finished processing in %v", idx, time.Since(t0))
		}(i, instance)
	}
	wg.Wait()

	log.Println("All requests processed.")
	log.Println("LLM inference system shutting down")
}

func init() {
	log.SetFlags(log.Lshortfile)
    log.SetPrefix("")
}
