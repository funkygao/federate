package main

import (
	"fmt"
	"log"
)

// PrefillInstance represents a prefill instance.
// It takes the user inputs to generate the first token: prefill phase, computation-bound.
type PrefillInstance struct {
	ID              string
	Scheduler       *LocalScheduler
	LengthPredictor *LengthPredictor
	LLMEngine       *LLMEngine
	Dispatcher      *Dispatcher
	Requests        []Request
}

func NewPrefillInstance(id int, policy SchedulingPolicy, clusterMonitor *ClusterMonitor) *PrefillInstance {
	return &PrefillInstance{
		ID:              fmt.Sprintf("prefill-%d", id),
		Scheduler:       NewLocalScheduler(policy),
		LengthPredictor: &LengthPredictor{BatchSize: 32},
		LLMEngine:       &LLMEngine{},
		Dispatcher:      NewDispatcher(clusterMonitor),
	}
}

func (pi *PrefillInstance) AddRequest(request Request) {
	pi.Requests = append(pi.Requests, request)
}

func (pi *PrefillInstance) GetLoad() int {
	return len(pi.Requests)
}

func (pi *PrefillInstance) ProcessRequests() {
	log.Printf("Prefill instance %s starting to process %d requests", pi.ID, len(pi.Requests))

	requests := pi.Requests
	pi.Requests = nil // Clear the requests

	// Schedule requests
	pi.Scheduler.ScheduleRequests(requests)
	scheduledRequests := pi.Scheduler.GetScheduledRequests()

	log.Printf("Prefill instance %s starting length prediction and LLM prefill", pi.ID)
	lengthPredictions := pi.LengthPredictor.PredictLength(scheduledRequests)
	prefillResults := pi.LLMEngine.ChunkedPrefill(scheduledRequests, lengthPredictions)
	log.Printf("Prefill instance %s finished length prediction and LLM prefill", pi.ID)

	log.Printf("Prefill instance %s dispatching %d results", pi.ID, len(prefillResults))
	for _, result := range prefillResults {
		pi.Dispatcher.DispatchRequest(result)
	}

	log.Printf("Prefill instance %s finished processing all requests", pi.ID)
}
