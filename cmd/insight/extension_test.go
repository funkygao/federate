package insight

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestIsExtensionInterface(t *testing.T) {
	tests := []struct {
		name     string
		code     string
		expected bool
	}{
		{
			name: "Valid extension interface with public keyword",
			code: `
package com.example;

public interface IExampleExt extends IDomainExtension {
    void someMethod();
}`,
			expected: true,
		},
		{
			name: "Valid extension interface without public keyword",
			code: `
package com.example;

interface IExampleExt extends IDomainExtension {
    void someMethod();
}`,
			expected: true,
		},
		{
			name: "AfterLocateOrderSplitterExt",
			code: `
package com.goog.wms.outbound.plan.locating.domain.ext;

import com.goog.wms.outbound.plan.locating.domain.vo.AfterLocateSplitOrderRequestVo;
import com.goog.wms.outbound.plan.locating.domain.vo.LocateSplitOrderGroupVo;
import io.github.dddplus.ext.IDomainExtension;

import java.util.List;
import java.util.Optional;

/**
 * @author someone
 * @date 2014/10/16 17:45
 */
public interface AfterLocateOrderSplitterExt extends IDomainExtension{
    /**
     * 定位后包裹拆单
     * @param requestVo
     * @return
     */
    Optional<List<LocateSplitOrderGroupVo>> splitOrder(AfterLocateSplitOrderRequestVo requestVo);
}`,
			expected: true,
		},
		{
			name: "Not an extension interface",
			code: `
package com.example;

public interface IExample {
    void someMethod();
}`,
			expected: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := interfaceRegex.MatchString(tt.code)
			assert.Equal(t, tt.expected, result, "isExtensionInterface() result not as expected")
		})
	}
}

func TestExtractMethods(t *testing.T) {
	tests := []struct {
		name     string
		code     string
		expected []Method
	}{
		{
			name: "Interface with one method",
			code: `
package com.example;

public interface IExampleExt extends IDomainExtension {
    /**
     * Some method description
     */
    void someMethod();
}`,
			expected: []Method{
				{Name: "someMethod", Javadoc: "Some method description", Dir: "foo"},
			},
		},
		{
			name: "AfterLocateOrderSplitterExt",
			code: `
package com.goog.wms.outbound.plan.locating.domain.ext;

import com.goog.wms.outbound.plan.locating.domain.vo.AfterLocateSplitOrderRequestVo;
import com.goog.wms.outbound.plan.locating.domain.vo.LocateSplitOrderGroupVo;
import io.github.dddplus.ext.IDomainExtension;

import java.util.List;
import java.util.Optional;

/**
 * @author someone
 * @date 2014/10/16 17:45
 */
public interface AfterLocateOrderSplitterExt extends IDomainExtension{
    /**
     * 定位后包裹拆单
     * @param requestVo
     * @return
     */
    Optional<List<LocateSplitOrderGroupVo>> splitOrder(AfterLocateSplitOrderRequestVo requestVo);
}`,
			expected: []Method{
				{Name: "splitOrder", Javadoc: "定位后包裹拆单", Dir: "foo"},
			},
		},
		{
			name: "Method has no javadoc",
			code: `
import java.util.List;

/**
 * @description
 * @created by me
 * @date 2024/3/6
 */
public interface IPrepareOrderExt extends IDomainExtension {

    void prepare(List<ShipmentOrder> shipmentOrders, ComposeContext composeContext);

}
`,
			expected: []Method{
				{Name: "prepare", Javadoc: "", Dir: "foo"},
			},
		},
		{
			name: "1",
			code: `
import java.math.BigDecimal;

/**
 * 箱子逻辑体积扩展
 *
 * @author zhouyiru
 * @date 2024/01/30
 */
public interface ICartonLogicalVolumeExt extends IDomainExtension {


    /**
     * 计算.箱子体积
     *
     * @param carton
     * @return {@link BigDecimal}
     */
    BigDecimal computeVolume(Carton carton);


    /**
     * 计算.箱子长宽高
     *
     * @param carton
     */
    Triple<BigDecimal, BigDecimal, BigDecimal> computeLengthWidthHeight(Carton carton);

}`,
			expected: []Method{
				{Name: "computeVolume", Javadoc: "计算.箱子体积", Dir: "foo"},
				{Name: "computeLengthWidthHeight", Javadoc: "计算.箱子长宽高", Dir: "foo"},
			},
		},
		{
			name: "2",
			code: `
public interface IClaimCheckTaskExt extends IDomainExtension {

    /**
     * 可以准备开始复核？
     *
     * @throws WmsOutboundException 不能的话 具体原因外显给用户
     */
    void canReadyToCheck(AnyCode anyCode, PlatformNo platformNo, WmsOperator operator, WarehouseNo warehouseNo) throws WmsOutboundException;
}                
`,
			expected: []Method{
				{Name: "canReadyToCheck", Javadoc: "可以准备开始复核？", Dir: "foo"},
			},
		},
		{
			name: "3",
			code: `
public interface IShippingBySelfExt extends IDomainExtension {


    /**
     * 根据「客户自提码」匹配订单
     *
     * @param selfCode
     * @return {@link OrderNo }
     * @apiNote 客户拿「自提码」来仓库直接提货
     */
    @ExtExecutionDoc(when = "客户需要自提，拿自提码到仓库提货，需要外部系统获取到订单返回WMS", effect = "客户自提")
    OrderNo matchOrderBySelfCode(String selfCode);
}
`,
			expected: []Method{
				{Name: "matchOrderBySelfCode", Javadoc: "根据「客户自提码」匹配订单", Dir: "foo"},
			},
		},
		{
			name: "4",
			code: `
public interface ISplitOrderDetailExt extends IDomainExtension {

    /**
     * 拆分订单明细
     *
     * com.jdwl.wms.taskassign.domain.order.entity.ShipmentOrderDetail#tagPick 标签拣选 是： true  否： false
     * com.jdwl.wms.taskassign.domain.order.entity.ShipmentOrderDetail#labelSplitType  参考： com.jdwl.wms.taskassign.support.enums.LabelSplitTypeEnum  本次如果拆
分拣货方式了，并且标签拣选的，应该返回 4
     * @param shipmentOrder 生产单
     */
    void splitOrderDetail(ShipmentOrder shipmentOrder, ComposeContext composeContext);
}`,
			expected: []Method{
				{Name: "splitOrderDetail", Javadoc: "拆分订单明细", Dir: "foo"},
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := extractMethods("test.java", tt.code, "foo")
			assert.Equal(t, tt.expected, result, "extractMethods() result not as expected")
		})
	}
}
