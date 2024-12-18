package main

import (
	"log"
	"sync"
)

// TCCParticipant interface defines the methods for a TCC-compatible resource
type TCCParticipant interface {
	Try() bool
	Confirm()
	Cancel()
}

// TCCCoordinator interface defines the method for performing a TCC transaction
type TCCCoordinator interface {
	PerformTCC([]TCCParticipant) bool
}

// ConcreteResource implements the TCCParticipant interface
type ConcreteResource struct {
	name      string
	available int
	reserved  int

	mu sync.Mutex
}

func (r *ConcreteResource) Try() bool {
	r.mu.Lock()
	defer r.mu.Unlock()

	if r.available > 0 {
		r.available--
		r.reserved++
		log.Printf("[Try] Reserved 1 from %s", r.name)
		return true
	}
	log.Printf("[Try] Failed to reserve from %s", r.name)
	return false
}

func (r *ConcreteResource) Confirm() {
	r.mu.Lock()
	defer r.mu.Unlock()

	r.reserved--
	log.Printf("[Confirm] Confirmed 1 from %s", r.name)
}

func (r *ConcreteResource) Cancel() {
	r.mu.Lock()
	defer r.mu.Unlock()

	r.available++
	r.reserved--
	log.Printf("[Cancel] Cancelled 1 from %s", r.name)
}

// ConcreteTCCCoordinator implements the TCCCoordinator interface
type ConcreteTCCCoordinator struct{}

func (c *ConcreteTCCCoordinator) PerformTCC(participants []TCCParticipant) bool {
	allSucceeded, tryResults := c.performTry(participants)

	if !allSucceeded {
		log.Println("[Try] Failed, initiating cancellation")
		c.performCancel(tryResults, participants)
		return false
	} else {
		log.Println("[Try] Succeeded, initiating confirmation")
		c.performConfirm(participants)
		return true
	}
}

func (c *ConcreteTCCCoordinator) performTry(participants []TCCParticipant) (allSucceeded bool, tryResults []bool) {
	tryResults = make([]bool, len(participants))
	var wg sync.WaitGroup
	for i, participant := range participants {
		wg.Add(1)
		go func(index int, p TCCParticipant) {
			defer wg.Done()
			tryResults[index] = p.Try()
		}(i, participant)
	}
	wg.Wait()

	// Check if all Try operations succeeded
	allSucceeded = true
	for _, tryOk := range tryResults {
		if !tryOk {
			allSucceeded = false
			break
		}
	}

	return
}

func (c *ConcreteTCCCoordinator) performConfirm(participants []TCCParticipant) {
	// Confirm phase
	for _, participant := range participants {
		participant.Confirm()
	}
}

func (c *ConcreteTCCCoordinator) performCancel(tryResults []bool, participants []TCCParticipant) {
	for i, tryOk := range tryResults {
		if tryOk {
			log.Printf("Call participants[%d] to cancel", i)
			participants[i].Cancel()
		}
	}
}

func main() {
	resourceA := &ConcreteResource{name: "Resource A", available: 2}
	resourceB := &ConcreteResource{name: "Resource B", available: 2}
	resourceC := &ConcreteResource{name: "Resource C", available: 1}

	coordinator := &ConcreteTCCCoordinator{}

	participants := []TCCParticipant{resourceA, resourceB, resourceC}

	success := coordinator.PerformTCC(participants)
	log.Printf("Transaction 1 success: %v", success)

	// Try another transaction
	success = coordinator.PerformTCC(participants)
	log.Printf("Transaction 2 success: %v", success)
}
