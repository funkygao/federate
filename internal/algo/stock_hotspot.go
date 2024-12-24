package main

import (
	"context"
	"database/sql"
	"errors"
	"time"

	"github.com/go-redis/redis/v8"
)

// 实际的预占库存逻辑可能远比这复杂，例如：RPC调用结合DB操作；一个sku有多条库存记录(一盘货)
// 需要在 lua 里重新再实现一遍
const scriptReserve = `
  local stock = redis.call('get', KEYS[1])
  if not stock then
      return {err = "NOT_FOUND"}
  end

  stock = tonumber(stock)
  if stock < tonumber(ARGV[1]) then
      return {err = "INSUFFICIENT_STOCK"}
  end

  -- 预占库存
  local new_stock = stock - tonumber(ARGV[1])
  redis.call('set', KEYS[1], new_stock)

  -- 记录库存流水
  local flow_key = KEYS[1] .. ":flow"
  local flow_id = redis.call('incr', flow_key .. ":id")
  redis.call('zadd', flow_key, flow_id, string.format("%d|%d|%d|%s", flow_id, os.time(), -quantity, ARGV[2]))
  return {ok = new_stock}
`

type Database interface {
	BeginTx(ctx context.Context, opts *sql.TxOptions) (*sql.Tx, error)
}

type Cache interface {
	Eval(script string, keys []string, args ...any) (any, error)
	Watch(fn func(*redis.Tx) error, keys ...string) error
}

type HotSpotRuleService interface {
	IsHotSpot(sku string) bool
}

type InventoryService struct {
	db    Database
	cache Cache

	hotSpot HotSpotRuleService
}

// 预占库存
func (s *InventoryService) ReserveInventory(ctx context.Context, sku string, quantity int) error {
	if s.hotSpot.IsHotSpot(sku) {
		return s.reserveInCache(ctx, sku, quantity)
	}

	return s.reserveInDB(ctx, sku, quantity)
}

func (s *InventoryService) reserveInCache(ctx context.Context, sku string, quantity int) (err error) {
	var result map[string]any
	// lua script 同时具备原子性和隔离性，而 redis 事务只有 EXEC 时才真正执行，之前操作可能被其他请求插队，不具备隔离性，才需要 Watch
	result, err = s.cache.Eval(scriptReserve, []string{sku}, quantity).Result()
	if err != nil {
		// redis crash !
		return
	}

	if errMsg, ok := result["err"]; ok {
		if errMsg == "NOT_FOUND" {
			// 规则变了，刚才是热点商品，现在变成非热点
			return s.reserveInDB(ctx, sku, quantity)
		}

		return errors.New(errMsg.(string))
	}

	return nil
}

func (s *InventoryService) reserveInDB(ctx context.Context, sku string, quantity int) error {
	tx, err := s.db.BeginTx(ctx, &sql.TxOptions{Isolation: sql.LevelRepeatableRead})
	if err != nil {
		return err
	}

	// Implement optimistic locking
	var stock int
	var version int
	err = tx.QueryRow("SELECT stock, version FROM inventory WHERE sku_id = ?", sku).Scan(&stock, &version)
	if err != nil {
		tx.Rollback()
		return err
	}

	if stock < quantity {
		tx.Rollback()
		return errors.New("insufficient stock")
	}

	newStock := stock - quantity
	if newStock < 0 {
		return errors.New("insufficient stock")
	}

	result, err := tx.Exec("UPDATE inventory SET stock = ?, version = version + 1 WHERE sku_id = ? AND version = ?", newStock, sku, version)
	if err != nil {
		tx.Rollback()
		return err
	}

	rowsAffected, err := result.RowsAffected()
	if err != nil {
		tx.Rollback()
		return err
	}

	if rowsAffected == 0 {
		// Optimistic lock failed, retry
		tx.Rollback()
		return s.reserveInDB(sku, quantity)
	}

	// 记录库存流水

	return tx.Commit()
}

func (s *HotSpotRuleServiceImpl) setHot(ctx context.Context, sku string) error {
	var stock int
	err := s.db.QueryRowContext(ctx, "SELECT stock FROM inventory WHERE sku_id = ?", sku).Scan(&stock)
	if err != nil {
		return err
	}

	// 使用 Redis 事务确保原子性
	_, err = s.cache.TxPipelined(ctx, func(pipe redis.Pipeliner) error {
		pipe.Set(ctx, sku, stock, 0)
		pipe.Set(ctx, sku+":hot", "1", 0)
		return nil
	})
	return err
}

func (s *HotSpotRuleServiceImpl) setCold(ctx context.Context, sku string) error {
	// 第一阶段：准备
	err := s.prepareSetCold(ctx, sku)
	if err != nil {
		return err
	}

	// 第二阶段：提交
	err = s.commitSetCold(ctx, sku)
	if err != nil {
		// 如果提交失败，尝试回滚
		s.rollbackSetCold(ctx, sku, stock)
		return err
	}
	return nil
}

func (s *HotSpotRuleServiceImpl) prepareSetCold(ctx context.Context, sku string) error {
	return s.cache.Watch(ctx, func(tx *redis.Tx) error {
		stock, err := tx.Get(ctx, sku).Result()
		if err != nil {
			return err
		}

		// tx.Get 与 tx.TxPipelined 之间，可能被其他请求插队，因此需要 Watch 乐观锁保证隔离性
		// 也可以换成 lua script，就不需要事务了

		// 标记为准备状态，包含更多信息
		_, err = tx.TxPipelined(ctx, func(pipe redis.Pipeliner) error {
			pipe.HSet(ctx, sku+":prepare_cold", map[string]any{
				"stock":     stock,
				"status":    "prepared",
				"timestamp": time.Now().Unix(),
			})
			pipe.Expire(ctx, sku+":prepare_cold", 10*time.Minute)
			return nil
		})
		return err
	}, sku)
}

func (s *HotSpotRuleServiceImpl) commitSetCold(ctx context.Context, sku string) error {
	// 获取准备阶段的信息
	prepareInfo, err := s.cache.HGetAll(ctx, sku+":prepare_cold").Result()
	if err != nil {
		return err
	}

	stock, _ := prepareInfo["stock"]

	// 更新 MySQL 中的库存
	_, err := s.db.ExecContext(ctx, "UPDATE inventory SET stock = ? WHERE sku_id = ?", stock, sku)
	if err != nil {
		return err
	}

	// 如果 MySQL 更新成功，更新准备状态并删除 Redis 中的数据
	_, err = s.cache.TxPipelined(ctx, func(pipe redis.Pipeliner) error {
		pipe.HSet(ctx, sku+":prepare_cold", "status", "committed")
		pipe.Del(ctx, sku)
		pipe.Del(ctx, sku+":hot")
		return nil
	})

	if err != nil {
		return err
	}

	// 最后删除准备状态的键
	return s.cache.Del(ctx, sku+":prepare_cold").Err()
}

func (s *HotSpotRuleServiceImpl) rollbackSetCold(ctx context.Context, sku, stock string) {
	// 删除准备状态的标记
	s.cache.Del(ctx, sku+":prepare_cold")
}

func (s *HotSpotRuleServiceImpl) recoverIncompleteOperations(ctx context.Context) {
	// 可以让数据库实现 scan
	keys, _ := s.cache.Keys(ctx, "*:prepare_cold").Result()
	for _, key := range keys {
		sku := strings.TrimSuffix(key, ":prepare_cold")
		info, _ := s.cache.HGetAll(ctx, key).Result()

		switch info["status"] {
		case "prepared":
			// 操作未完成，回滚
			s.rollbackSetCold(ctx, sku)
		case "committed":
			// MySQL 已更新，但 Redis 数据未完全删除，完成删除操作
			s.cache.Del(ctx, sku, sku+":hot", key)
		}
	}
}
