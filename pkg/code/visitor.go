package code

// JavaFileVisitor defines the interface for visitors that can analyze Java files.
type JavaFileVisitor interface {
	Visit(jf *JavaFile)
}
