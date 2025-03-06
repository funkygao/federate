import sys																	
import os																	
import time																	
import datetime																	
sys.path.append(os.getenv('HIVE_TASK'))																	
from HiveTask import HiveTask																	

ht = HiveTask()																	
business_date = ht.data_day_str																	
today = datetime.datetime.now().strftime('%Y-%m-%d')																	
day_before_1_str=(datetime.datetime.now() + datetime.timedelta(-1)).strftime('%Y-%m-%d')																	
today_now = datetime.datetime.now()																	
five_today = today_now.replace(hour=5, minute=0, second=0, microsecond=0)																	
five_today_timestamp = str(time.mktime(five_today.timetuple()))																	
																	
sql = """
use adm;
set mapred.output.compress=true;
set mapred.job.priority=VERY_HIGH;
set hive.exec.compress.output=true;
set hive.exec.parallel=true;
set hive.exec.dynamic.partition=true;
set hive.exec.dynamic.partition.mode=nonstrict;
set hive.exec.max.dynamic.partitions=100000;
set hive.exec.max.dynamic.partitions.pernode=100000;
set hive.default.fileformat=Orc;
set hive.ppd.remove.duplicatefilters=false;
set hive.auto.convert.join=true;

INSERT overwrite table app.wms_stock_with_lot_det_d partition(dt='""" + today + """',pt_source='6-d')
SELECT
	st_tmp.wms_org_no as org_no,--预留机构号
	(CASE WHEN st_tmp.warehouse_no like '%_%' THEN split(st_tmp.warehouse_no,'_')[0] ELSE st_tmp.wms_distribution_no END) as distribute_no,--配送中心
	(CASE WHEN st_tmp.warehouse_no like '%_%' THEN split(st_tmp.warehouse_no,'_')[1] ELSE st_tmp.warehouse_no END) as warehouse_no,--仓编号
	st_tmp.mini_warehouse_no as site_no,--站点编号
	st_tmp.sku as goods_no,--商品编码
	'' as goods_name,--商品名称
	'' as goods_level,--预留等级
	st_tmp.lot_no,--批次号
	st_tmp.stock_qty - IFNULL(in_tmp.qty,0) as stock_num,--库存量
	st_tmp.owner_no,--货主
	DATE '""" + today + """' as dt_date,--日期
	(CASE WHEN st_tmp.warehouse_no in ('347_93','10774_93') THEN '4' else '6' END) as source,--来源
	int('"""+five_today_timestamp+"""') as ts_timestamp--时间戳
FROM
	(
		SELECT
			first(st.source_ip) as source_ip,
            first(st.source_db) as source_db,
            first(st.source_tb) as source_tb,
            first(st.snapshot_date) as snapshot_date,
            first(st.tenant_code) as tenant_code,
			first(st.owner_no) as owner_no,
            st.warehouse_no,
            st.sku,
            st.lot_no,
            sum(st.stock_qty) as stock_qty,
            sum(st.prepicked_qty) prepicked_qty,
            sum(st.premoved_qty) premoved_qty,
            sum(st.frozen_qty) frozen_qty,
            sum(st.diff_qty) diff_qty,
            sum(st.broken_qty) broken_qty,
			first(wc.wms_distribution_no) as wms_distribution_no,
			first(wc.wms_org_no) as wms_org_no,
			first(wc.mini_warehouse_no) as mini_warehouse_no
		FROM
			fdm.wms_stock_st_stock_snapshot_st st 
		LEFT JOIN fdm.master_wms_six_bind_mini_chain wc 
            ON st.tenant_code = wc.wms_tenant_no and st.warehouse_no = concat(wc.wms_distribution_no,'_',wc.wms_warehouse_no) and wc.dp = 'ACTIVE'
		WHERE 
            st.dt = '"""+day_before_1_str+"""' 
            AND st.snapshot_date = '"""+today+"""' 
            AND st.tenant_code not in('TC65926780') 
            AND st.warehouse_no not like '0_%' 
            AND st.sku not like 'EMG%' 
            AND st.sku not like 'CMG%' 
            AND st.sku not like 'ERP%' 
            AND st.v_deleted = 0
		    AND wc.mini_warehouse_no not in('2097753','2097751')
		GROUP BY st.warehouse_no, st.sku, st.lot_no
	)
	st_tmp
	LEFT JOIN 
	(
		SELECT
			warehouse_no,
			sku,
			lot_no,
			sum(qty) AS qty
		FROM
			fdm.wms_inbound_ib_transit_stock_snapshot_st
		WHERE
			dt = '"""+day_before_1_str+"""'
			AND snapshot_date = '"""+today+"""'
			and warehouse_no not like '0_%' 
			and sku not like 'EMG%' and sku not like 'CMG%'
			AND v_deleted = 0
		GROUP BY
			warehouse_no, sku, lot_no
	)	
	in_tmp 
    ON st_tmp.warehouse_no = in_tmp.warehouse_no and st_tmp.sku = in_tmp.sku and st_tmp.lot_no = in_tmp.lot_no
"""																	
																	
ht.exec_sql(																	
    schema_name='app',																	
    table_name='wms_stock_with_lot_det_d',																	
    sql=sql,																	
    merge_flag=True,																	
    merge_part_dir=['dt=' + today+'/pt_source=6-d'])																	
