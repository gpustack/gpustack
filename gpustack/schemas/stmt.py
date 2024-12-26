worker_after_drop_view_stmt_sqlite = "DROP VIEW IF EXISTS gpu_devices_view"
worker_after_create_view_stmt_sqlite = """
CREATE VIEW IF NOT EXISTS gpu_devices_view AS
SELECT
    w.name || ':' || json_extract(value, '$.type') || ':' || json_extract(value, '$.index') AS id,
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
    json_extract(value, '$.temperature') AS temperature,
    json_extract(value, '$.labels') AS labels,
    json_extract(value, '$.type') AS type
FROM
    workers w,
    json_each(w.status, '$.gpu_devices')
WHERE
    json_array_length(w.status, '$.gpu_devices') > 0
"""

worker_after_drop_view_stmt_postgres = "DROP VIEW IF EXISTS gpu_devices_view CASCADE"
worker_after_create_view_stmt_postgres = """
CREATE VIEW gpu_devices_view AS
SELECT
    w.name || ':' || (gpu_device::json->>'type') || ':' || (gpu_device::json->>'index') AS id,
    w.id as worker_id,
    w.name as worker_name,
    w.ip as worker_ip,
    w.created_at,
    w.updated_at,
    w.deleted_at,
    (gpu_device::json->>'uuid') AS uuid,
    (gpu_device::json->>'name') AS name,
    (gpu_device::json->>'vendor') AS vendor,
    (gpu_device::json->>'index')::INTEGER AS "index",
    (gpu_device::json->>'core')::JSONB AS core,
    (gpu_device::json->>'memory')::JSONB AS memory,
    (gpu_device::json->>'temperature')::FLOAT AS temperature,
    (gpu_device::json->>'labels')::JSONB AS labels,
    (gpu_device::json->>'type') AS type
FROM
    workers w,
    LATERAL json_array_elements(w.status::json->'gpu_devices') AS gpu_device
WHERE
    json_array_length(w.status::json->'gpu_devices') > 0
"""
