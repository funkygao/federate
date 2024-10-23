package concurrent

import (
	"errors"
	"sync"
	"testing"
	"time"
)

// mockTask is a simple task that can be configured to succeed or fail
type mockTask struct {
	shouldFail bool
	delay      time.Duration
}

func (t *mockTask) Execute() error {
	time.Sleep(t.delay)
	if t.shouldFail {
		return errors.New("task failed")
	}
	return nil
}

func TestParallelExecutor(t *testing.T) {
	tests := []struct {
		name           string
		maxWorkers     int
		tasks          []Task
		expectedErrors int
	}{
		{
			name:           "All tasks succeed",
			maxWorkers:     2,
			tasks:          []Task{&mockTask{}, &mockTask{}, &mockTask{}},
			expectedErrors: 0,
		},
		{
			name:           "Some tasks fail",
			maxWorkers:     2,
			tasks:          []Task{&mockTask{}, &mockTask{shouldFail: true}, &mockTask{shouldFail: true}},
			expectedErrors: 2,
		},
		{
			name:           "Max workers respected",
			maxWorkers:     2,
			tasks:          []Task{&mockTask{delay: 100 * time.Millisecond}, &mockTask{delay: 100 * time.Millisecond}, &mockTask{delay: 100 * time.Millisecond}},
			expectedErrors: 0,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			executor := NewParallelExecutor(tt.maxWorkers)
			for _, task := range tt.tasks {
				executor.AddTask(task)
			}

			start := time.Now()
			errors := executor.Execute()
			duration := time.Since(start)

			if len(errors) != tt.expectedErrors {
				t.Errorf("Expected %d errors, got %d", tt.expectedErrors, len(errors))
			}

			if tt.name == "Max workers respected" {
				expectedDuration := 200 * time.Millisecond
				if duration < expectedDuration {
					t.Errorf("Expected duration to be at least %v, got %v", expectedDuration, duration)
				}
			}
		})
	}
}

func TestCounter(t *testing.T) {
	counter := NewCounter()
	var wg sync.WaitGroup
	numGoroutines := 100
	incrementsPerGoroutine := 1000

	for i := 0; i < numGoroutines; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for j := 0; j < incrementsPerGoroutine; j++ {
				counter.Increment()
			}
		}()
	}

	wg.Wait()

	expected := int64(numGoroutines * incrementsPerGoroutine)
	if counter.Value() != expected {
		t.Errorf("Expected counter value to be %d, got %d", expected, counter.Value())
	}
}
