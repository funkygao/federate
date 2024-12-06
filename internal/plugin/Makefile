all:demo

fmt:
	go fmt ./...

clean:
	rm -f *.so

demo:fmt ## Build  the demo plugin
	go build -buildmode=plugin -o demo_plugin.so demo_plugin.go


