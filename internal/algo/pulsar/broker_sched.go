package main

import (
	"log"
	"time"
)

func (b *broker) processDelayedMessages() {
	ticker := time.NewTicker(100 * time.Millisecond)
	defer ticker.Stop()

	for {
		select {
		case <-ticker.C:
			// 一个时间周期内处理尽可能多的延迟消息
			for {
				msg, hasDueMsg := b.delayQueue.Poll()
				if !hasDueMsg {
					break
				}

				// 消息已经准备好，发布到相应的主题
				log.Printf("%s delay message is due: %v", b.logIdent(), msg)
				if err := b.Publish(msg); err != nil {
					log.Printf("Error publishing delayed message: %v", err)
				}
			}
		}
	}
}

func (b *broker) manageLedgerRetention() {
	ticker := time.NewTicker(1 * time.Hour)
	defer ticker.Stop()

	for {
		select {
		case <-ticker.C:
			b.cleanupExpiredLedgers()
		}
	}
}

func (b *broker) cleanupExpiredLedgers() {
	b.mu.Lock()
	defer b.mu.Unlock()

	retentionMaxAge := 48 * time.Hour
	cutoffTime := time.Now().Add(retentionMaxAge)
	log.Printf("%s Starting ledger retention cleanup, cutoff time: %v", b.logIdent(), cutoffTime)

	for _, topic := range b.topics {
		for _, partition := range topic.Partitions {
			var activeLedgers []LedgerID
			for _, ledgerID := range partition.Ledgers {
				ledger, err := b.bkClient.OpenLedger(ledgerID)
				if err != nil {
					log.Printf("%s Failed to open ledger %d for retention check: %v", b.logIdent(), ledgerID, err)
					continue
				}

				if ledger.Age() > retentionMaxAge {
					ledger.Close()
					b.bkClient.DeleteLedger(ledgerID)
				} else {
					activeLedgers = append(activeLedgers, ledgerID)
				}
			}

			partition.Ledgers = activeLedgers
		}
	}
}

func (b *broker) rebalance() {
}
