package concurrent

// Task represents a unit of work to be done
type Task interface {
	Execute() error
}

// Result represents the result of a task execution
type Result interface{}
