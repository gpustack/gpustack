worker_after_drop_view_stmt = "DROP VIEW IF EXISTS gpu_devices_view"
worker_after_create_view_stmt = """
CREATE VIEW IF NOT EXISTS gpu_devices_view AS
SELECT
    w.name || '-' || json_extract(value, '$.name') || '-' || json_extract(value, '$.index') AS id,
    w.id as worker_id,
    w.name as worker_name,
    w.ip as worker_ip,
    w.created_at,
    w.updated_at,
    w.deleted_at,
    json_extract(value, '$.uuid') AS uuid,
    json_extract(value, '$.name') AS name,
    json_extract(value, '$.vendor') AS vendor,
    json_extract(value, '$.index') AS 'index',
    json_extract(value, '$.core') AS core,
    json_extract(value, '$.memory') AS memory,
    json_extract(value, '$.temperature') AS temperature
FROM
    workers w,
    json_each(w.status, '$.gpu_devices')
WHERE
    json_array_length(w.status, '$.gpu_devices') > 0
"""
