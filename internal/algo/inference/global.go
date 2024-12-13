package main

import (
	"log"
	"time"
)

// RequestStatus represents the current status of a request
type RequestStatus struct {
	ID              string
	ArrivalTime     time.Time
	Phase           string // "prefill" or "decode"
	SLA             time.Duration
	PrefillInstance *PrefillInstance
	DecodeInstance  *DecodeInstance
}

// GlobalScheduler manages request distribution and status tracking
type GlobalScheduler struct {
	PrefillInstances []*PrefillInstance
	RequestTable     map[string]*RequestStatus
	ClusterMonitor   *ClusterMonitor
}

func NewGlobalScheduler(numInstances int, clusterMonitor *ClusterMonitor) *GlobalScheduler {
	gs := &GlobalScheduler{
		PrefillInstances: make([]*PrefillInstance, numInstances),
		RequestTable:     make(map[string]*RequestStatus),
		ClusterMonitor:   clusterMonitor,
	}
	for i := 0; i < numInstances; i++ {
		gs.PrefillInstances[i] = NewPrefillInstance(i, SJF, clusterMonitor)
	}
	return gs
}

func (gs *GlobalScheduler) HandleIncomingRequest(request Request) {
	prefillInstance := gs.chooseLeastLoadedPrefillInstance()
	log.Printf("Request[%s] with %d prompt tokens dispatched to prefill instance: %s", request.ID, request.PromptTokens, prefillInstance.ID)

	status := &RequestStatus{
		ID:              request.ID,
		ArrivalTime:     time.Now(),
		Phase:           "prefill",
		SLA:             request.SLA,
		PrefillInstance: prefillInstance,
	}
	gs.RequestTable[request.ID] = status

	prefillInstance.AddRequest(request)
}

func (gs *GlobalScheduler) chooseLeastLoadedPrefillInstance() *PrefillInstance {
	var leastLoaded *PrefillInstance
	minLoad := int(^uint(0) >> 1) // Max int

	for _, instance := range gs.PrefillInstances {
		load := instance.GetLoad()
		if load < minLoad {
			minLoad = load
			leastLoaded = instance
		}
	}
	return leastLoaded
}

func (gs *GlobalScheduler) UpdateRequestStatus(requestID string, phase string, decodeInstance *DecodeInstance) {
	if status, exists := gs.RequestTable[requestID]; exists {
		status.Phase = phase
		status.DecodeInstance = decodeInstance
		log.Printf("Updating status for request %s: Phase changed from %s to %s, Decode Instance: %s",
			requestID, status.Phase, phase, decodeInstance.ID)
	}
}

// ClusterMonitor represents the cluster monitor that broadcasts decode instances' load information
type ClusterMonitor struct {
	DecodeInstances []*DecodeInstance
}

func NewClusterMonitor() *ClusterMonitor {
	return &ClusterMonitor{
		DecodeInstances: make([]*DecodeInstance, 0),
	}
}

func (cm *ClusterMonitor) AddDecodeInstance(instance *DecodeInstance) {
	cm.DecodeInstances = append(cm.DecodeInstances, instance)
}

func (cm *ClusterMonitor) UpdateInstanceLoad(instance *DecodeInstance) {
	for i, di := range cm.DecodeInstances {
		if di.ID == instance.ID {
			cm.DecodeInstances[i] = instance
			break
		}
	}
}

func (cm *ClusterMonitor) GetDecodeInstances() []*DecodeInstance {
	return cm.DecodeInstances
}
