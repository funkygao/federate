package main

import (
	"fmt"
	"log"
	"sync"
)

// SagaStep interface defines the methods for a saga step
type SagaStep interface {
	Execute() error
	Compensate() error

	GetName() string
}

// Saga interface defines the methods for a saga
type Saga interface {
	AddStep(step SagaStep)
	Execute() error
}

// ConcreteStep implements the SagaStep interface
type ConcreteStep struct {
	name       string
	execute    func() error
	compensate func() error
}

func (s *ConcreteStep) Execute() error {
	return s.execute()
}

func (s *ConcreteStep) Compensate() error {
	return s.compensate()
}

func (s *ConcreteStep) GetName() string {
	return s.name
}

// ConcreteSaga implements the Saga interface
type ConcreteSaga struct {
	steps []SagaStep
	mu    sync.Mutex
}

func (s *ConcreteSaga) AddStep(step SagaStep) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.steps = append(s.steps, step)
}

func (s *ConcreteSaga) Execute() error {
	executedSteps := []SagaStep{}

	for _, step := range s.steps {
		log.Printf("Executing step: %s", step.GetName())
		err := step.Execute()
		if err != nil {
			log.Printf("Step %s failed: %v", step.GetName(), err)
			return s.compensate(executedSteps)
		}
		executedSteps = append(executedSteps, step)
	}

	log.Println("All steps executed successfully")
	return nil
}

func (s *ConcreteSaga) compensate(executedSteps []SagaStep) error {
	log.Println("Starting compensation")
	var compensationError error

	for i := len(executedSteps) - 1; i >= 0; i-- {
		step := executedSteps[i]
		log.Printf("Compensating step: %s", step.GetName())
		err := step.Compensate()
		if err != nil {
			log.Printf("Compensation failed for step %s: %v", step.GetName(), err)
			if compensationError == nil {
				compensationError = fmt.Errorf("compensation failed for step %s", step.GetName())
			}
		}
	}

	if compensationError != nil {
		return fmt.Errorf("saga failed and compensation had errors: %v", compensationError)
	}

	log.Println("Compensation completed successfully")
	return fmt.Errorf("saga failed but was successfully compensated")
}

// SagaCoordinator manages the execution of sagas
type SagaCoordinator struct {
	sagas []*ConcreteSaga
}

func NewSagaCoordinator() *SagaCoordinator {
	return &SagaCoordinator{
		sagas: make([]*ConcreteSaga, 0),
	}
}

func (sc *SagaCoordinator) CreateSaga() *ConcreteSaga {
	saga := &ConcreteSaga{}
	sc.sagas = append(sc.sagas, saga)
	return saga
}

func (sc *SagaCoordinator) ExecuteAll() []error {
	var errors []error
	for _, saga := range sc.sagas {
		if err := saga.Execute(); err != nil {
			errors = append(errors, err)
		}
	}
	return errors
}

func main() {
	coordinator := NewSagaCoordinator()

	// Create a saga for booking a trip
	tripBookingSaga := coordinator.CreateSaga()

	// Add steps to the saga
	tripBookingSaga.AddStep(&ConcreteStep{
		name: "Reserve Hotel",
		execute: func() error {
			log.Println("Reserving hotel")
			return nil
		},
		compensate: func() error {
			log.Println("Cancelling hotel reservation")
			return nil
		},
	})

	tripBookingSaga.AddStep(&ConcreteStep{
		name: "Book Flight",
		execute: func() error {
			log.Println("Booking flight")
			// Simulate a failure
			return fmt.Errorf("flight unavailable")
		},
		compensate: func() error {
			log.Println("Cancelling flight booking")
			return nil
		},
	})

	tripBookingSaga.AddStep(&ConcreteStep{
		name: "Charge Credit Card",
		execute: func() error {
			log.Println("Charging credit card")
			return nil
		},
		compensate: func() error {
			log.Println("Refunding credit card")
			return nil
		},
	})

	// Create another saga for a different operation
	inventoryUpdateSaga := coordinator.CreateSaga()

	inventoryUpdateSaga.AddStep(&ConcreteStep{
		name: "Update Inventory",
		execute: func() error {
			log.Println("Updating inventory")
			return nil
		},
		compensate: func() error {
			log.Println("Reverting inventory update")
			return nil
		},
	})

	// Execute all sagas
	errors := coordinator.ExecuteAll()

	// Check for errors
	if len(errors) > 0 {
		log.Println("Some sagas failed:")
		for _, err := range errors {
			log.Printf("- %v", err)
		}
	} else {
		log.Println("All sagas completed successfully")
	}
}
