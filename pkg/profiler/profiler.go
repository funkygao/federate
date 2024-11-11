package profiler

import (
	"log"
	"net/http"
	_ "net/http/pprof"
)

const PPROF_PORT = "9087"

func Enable() {
	go func() {
		log.Println(http.ListenAndServe("localhost:"+PPROF_PORT, nil))
	}()
}
