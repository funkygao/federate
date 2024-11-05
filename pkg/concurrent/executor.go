package concurrent

import (
	"log"
	"sync"
)

// ParallelExecutor handles parallel execution of tasks
type ParallelExecutor struct {
	maxWorkers int
	tasks      []Task
	name       string
}

// NewParallelExecutor creates a new ParallelExecutor
func NewParallelExecutor(maxWorkers int) *ParallelExecutor {
	return &ParallelExecutor{
		maxWorkers: maxWorkers,
		tasks:      []Task{},
	}
}

// AddTask adds a task to the executor
func (pe *ParallelExecutor) AddTask(task Task) {
	pe.tasks = append(pe.tasks, task)
}

func (pe *ParallelExecutor) SetName(name string) {
	pe.name = name
}

func (pe *ParallelExecutor) Tasks() []Task {
	return pe.tasks
}

// Execute runs all tasks in parallel and returns any errors
func (pe *ParallelExecutor) Execute() []error {
	var wg sync.WaitGroup
	errChan := make(chan error, len(pe.tasks))

	// Create a buffered channel to limit the number of concurrent goroutines
	semaphore := make(chan struct{}, pe.maxWorkers)

	if pe.name != "" {
		log.Printf("Excutor with %d tasks started: %s", len(pe.tasks), pe.name)
	}

	for _, task := range pe.tasks {
		wg.Add(1)
		go func(t Task) {
			defer wg.Done()
			semaphore <- struct{}{}        // Acquire a token
			defer func() { <-semaphore }() // Release the token

			if err := t.Execute(); err != nil {
				errChan <- err
			}
		}(task)
	}

	go func() {
		wg.Wait()
		close(errChan)
	}()

	var errors []error
	for err := range errChan {
		errors = append(errors, err)
	}

	return errors
}
