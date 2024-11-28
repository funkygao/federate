package code

import (
	"context"
)

// JavaFileVisitor defines the interface for visitors that can analyze Java files.
type JavaFileVisitor interface {
	Visit(ctx context.Context, jf *JavaFile)
}

// AcceptOption is a function type for optional parameters in Accept.
type AcceptOption func(*acceptOptions)

// acceptOptions holds the options for Accept.
type acceptOptions struct {
	ctx context.Context
}

// WithContext is an option to provide a context to Acceptor.
func WithContext(ctx context.Context) AcceptOption {
	return func(o *acceptOptions) {
		o.ctx = ctx
	}
}

type Acceptor interface {
	Accept(opts ...AcceptOption)
}

type Walker interface {
	Walk(opts ...AcceptOption) error
}

var (
	_ Acceptor = &JavaFile{}
	_ Walker   = &JavaWalker{}
)
