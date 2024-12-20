package javast

import (
	"federate/pkg/step"
)

// Execute instrumentation for the backlogs
func Instrument(bar step.Bar) error {
	driver := NewJavastDriver()
	return driver.Invoke(bar, backlog...)
}
