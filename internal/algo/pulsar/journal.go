package main

import (
	"encoding/binary"
	"fmt"
	"io"
	"log"
	"os"
	"path/filepath"
	"sync/atomic"
	"time"
)

type Journal interface {
	Append(entry JournalEntry) error
	Read(startPosition int64, maxEntries int) ([]JournalEntry, error)
	Sync() error
	Close() error

	Checkpoint() error
	Recover() error
}

type JournalEntry struct {
	LedgerID LedgerID
	EntryID  EntryID
	Data     Payload
}

func (entry *JournalEntry) HeaderSize() int {
	return 20
}

func (entry *JournalEntry) TotalSize() int {
	return entry.HeaderSize() + len(entry.Data)
}

func (entry *JournalEntry) Marshal() []byte {
	buf := make([]byte, entry.TotalSize())
	binary.BigEndian.PutUint64(buf[0:8], uint64(entry.LedgerID))
	binary.BigEndian.PutUint64(buf[8:16], uint64(entry.EntryID))
	binary.BigEndian.PutUint32(buf[16:20], uint32(len(entry.Data)))
	copy(buf[entry.HeaderSize():], entry.Data)
	return buf
}

func (entry *JournalEntry) UnmarshalFrom(r io.ReaderAt, position *int64) error {
	// header
	header := make([]byte, entry.HeaderSize())
	if _, err := r.ReadAt(header, *position); err != nil {
		return err
	}

	entry.LedgerID = LedgerID(binary.BigEndian.Uint64(header[0:8]))
	entry.EntryID = EntryID(binary.BigEndian.Uint64(header[8:16]))

	// body
	dataLen := binary.BigEndian.Uint32(header[16:20])
	entry.Data = make([]byte, dataLen)
	if _, err := r.ReadAt(entry.Data, *position+int64(entry.HeaderSize())); err != nil {
		return err
	}

	*position += int64(entry.TotalSize())
	return nil
}

type FileJournal struct {
	dir            string
	w              *os.File
	r              *os.File
	position       int64
	checkpointFile string
	stopChan       chan struct{}

	b Bookie
}

func NewFileJournal(dir string, b Bookie) (Journal, error) {
	// Ensure the directory exists
	dir = filepath.Join(dir, fmt.Sprintf("%d", b.ID()))
	if err := os.MkdirAll(dir, 0755); err != nil {
		return nil, err
	}

	wPath := filepath.Join(dir, "journal.data")
	w, err := os.OpenFile(wPath, os.O_WRONLY|os.O_APPEND|os.O_CREATE, 0666)
	if err != nil {
		return nil, err
	}

	r, err := os.Open(wPath)
	if err != nil {
		w.Close()
		return nil, err
	}

	info, err := w.Stat()
	if err != nil {
		w.Close()
		r.Close()
		return nil, err
	}

	j := &FileJournal{
		b:              b,
		dir:            dir,
		w:              w,
		r:              r,
		position:       info.Size(),
		checkpointFile: filepath.Join(dir, "journal.checkpoint"),
		stopChan:       make(chan struct{}),
	}

	log.Printf("Bookie[%d] Initial journal path: %s, position: %d", j.b.ID(), dir, j.position)

	if err = j.Recover(); err != nil {
		w.Close()
		r.Close()
		return nil, err
	}

	go j.checkpointManager()

	return j, nil
}

func (j *FileJournal) Append(entry JournalEntry) error {
	n, err := j.w.Write(entry.Marshal())
	if err != nil {
		return err
	}

	newPosition := atomic.AddInt64(&j.position, int64(n))
	log.Printf("Bookie[%d] Appended entry. New position: %d", j.b.ID(), newPosition)
	return nil
}

func (j *FileJournal) Read(startPosition int64, maxEntries int) ([]JournalEntry, error) {
	log.Printf("Bookie[%d] Reading from position %d, max entries: %d", j.b.ID(), startPosition, maxEntries)

	entries := make([]JournalEntry, 0, maxEntries)
	position := startPosition

	for i := 0; i < maxEntries; i++ {
		if position >= atomic.LoadInt64(&j.position) {
			log.Printf("Bookie[%d] Reached end of journal at position %d", j.b.ID(), position)
			break
		}

		var entry JournalEntry
		if err := entry.UnmarshalFrom(j.r, &position); err != nil {
			log.Printf("Bookie[%d] Error reading data at position %d: %v", j.b.ID(), position, err)
			return entries, err
		}

		entries = append(entries, entry)
	}

	log.Printf("Bookie[%d] Read %d entries", j.b.ID(), len(entries))
	return entries, nil
}

func (j *FileJournal) Sync() error {
	return j.w.Sync()
}

func (j *FileJournal) Close() error {
	close(j.stopChan)

	err1 := j.w.Close()
	err2 := j.r.Close()
	if err1 != nil {
		return err1
	}
	if err2 != nil {
		return err2
	}
	return nil
}

func (j *FileJournal) Checkpoint() error {
	log.Println("Bookie[%d] Creating checkpoint", j.b.ID())

	tempFile := j.checkpointFile + ".tmp"
	f, err := os.Create(tempFile)
	if err != nil {
		return err
	}
	defer f.Close()

	currentPosition := atomic.LoadInt64(&j.position)
	err = binary.Write(f, binary.BigEndian, currentPosition)
	if err != nil {
		return err
	}

	if err = f.Sync(); err != nil {
		return err
	}

	if err = os.Rename(tempFile, j.checkpointFile); err != nil {
		return err
	}

	log.Printf("Bookie[%d] Checkpoint created at position %d", j.b.ID(), currentPosition)
	return nil
}

func (j *FileJournal) Recover() error {
	log.Printf("Bookie[%d] Starting recovery process", j.b.ID())

	f, err := os.Open(j.checkpointFile)
	if os.IsNotExist(err) {
		return nil
	} else if err != nil {
		return err
	}
	defer f.Close()

	var checkpointPosition int64
	err = binary.Read(f, binary.BigEndian, &checkpointPosition)
	if err != nil {
		return err
	}

	fileInfo, err := j.w.Stat()
	if err != nil {
		return err
	}

	if checkpointPosition > fileInfo.Size() {
		return fmt.Errorf("invalid checkpoint: position %d exceeds file size %d", checkpointPosition, fileInfo.Size())
	}

	_, err = j.w.Seek(checkpointPosition, 0)
	if err != nil {
		return err
	}

	atomic.StoreInt64(&j.position, checkpointPosition)
	log.Printf("Bookie[%d] Recovered to position %d", j.b.ID(), checkpointPosition)
	return nil
}

func (j *FileJournal) checkpointManager() {
	ticker := time.NewTicker(5 * time.Second)
	defer ticker.Stop()

	for {
		select {
		case <-ticker.C:
			if err := j.Checkpoint(); err != nil {
				log.Printf("Bookie[%d] Error during scheduled checkpoint: %v", j.b.ID(), err)
			}
		case <-j.stopChan:
			return
		}
	}
}
