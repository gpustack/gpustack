"""add model categories

Revision ID: e6bf9e067296
Revises: 004c73a5c09e
Create Date: 2024-12-26 14:09:59.306468

"""
import json
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
import sqlmodel
import gpustack


# revision identifiers, used by Alembic.
revision: str = 'e6bf9e067296'
down_revision: Union[str, None] = '004c73a5c09e'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    with op.batch_alter_table('models', schema=None) as batch_op:
        batch_op.add_column(sa.Column('categories', sa.JSON(), nullable=True))
        batch_op.add_column(sa.Column('meta', sa.JSON(), nullable=True, default={}))

    with op.batch_alter_table('workers', schema=None) as batch_op:
        batch_op.add_column(sa.Column('port', sa.Integer(),
                            nullable=False, server_default="10150"))
        batch_op.add_column(sa.Column('unreachable', sa.Boolean(),
                            nullable=False, server_default="0"))
        batch_op.add_column(
            sa.Column('heartbeat_time', gpustack.schemas.common.UTCDateTime(), nullable=True))

    conn = op.get_bind()
    if conn.dialect.name == 'postgresql':
        # model_instance_state_enum
        existing_model_instance_state_enum_values = conn.execute(
            sa.text("SELECT unnest(enum_range(NULL::modelinstancestateenum))::text")
        ).fetchall()
        existing_model_instance_state_enum_values = [
            row[0] for row in existing_model_instance_state_enum_values]

        if 'STARTING' not in existing_model_instance_state_enum_values:
            conn.execute(sa.text("ALTER TYPE modelinstancestateenum ADD VALUE 'STARTING'"))

        if 'UNREACHABLE' not in existing_model_instance_state_enum_values:
            conn.execute(
                sa.text("ALTER TYPE modelinstancestateenum ADD VALUE 'UNREACHABLE'"))

        # worker_state_enum
        existing_worker_state_enum_values = conn.execute(
            sa.text("SELECT unnest(enum_range(NULL::workerstateenum))::text")
        ).fetchall()
        existing_worker_state_enum_values = [row[0]
                                             for row in existing_worker_state_enum_values]

        if 'UNREACHABLE' not in existing_worker_state_enum_values:
            conn.execute(
                sa.text("ALTER TYPE workerstateenum ADD VALUE 'UNREACHABLE'"))

        # model_usage_operation_enum
        existing_model_usage_operation_enum_values = conn.execute(
            sa.text("SELECT unnest(enum_range(NULL::operationenum))::text")
        ).fetchall()
        existing_model_usage_operation_enum_values = [row[0]
                                                      for row in existing_model_usage_operation_enum_values]
        if 'AUDIO_TRANSCRIPTION' not in existing_model_usage_operation_enum_values:
            conn.execute(
                sa.text("ALTER TYPE operationenum ADD VALUE 'AUDIO_TRANSCRIPTION'"))
        if 'AUDIO_SPEECH' not in existing_model_usage_operation_enum_values:
            conn.execute(
                sa.text("ALTER TYPE operationenum ADD VALUE 'AUDIO_SPEECH'"))

        conn.execute(
            sa.text("""
                UPDATE workers
                SET status = (
                    SELECT jsonb_set(
                        status::jsonb,
                        '{gpu_devices}',
                        (
                            SELECT jsonb_agg(
                                jsonb_set(
                                    jsonb_set(
                                        gpu_device,
                                        '{labels}',
                                        '{}'::jsonb
                                    ),
                                    '{type}',
                                    CASE
                                        WHEN gpu_device->>'vendor' = 'NVIDIA' THEN '"cuda"'
                                        WHEN gpu_device->>'vendor' = 'Moore Threads' THEN '"musa"'
                                        WHEN gpu_device->>'vendor' = 'Apple' THEN '"mps"'
                                        WHEN gpu_device->>'vendor' = 'Huawei' THEN '"npu"'
                                        WHEN gpu_device->>'vendor' = 'AMD' THEN '"rocm"'
                                        ELSE '"unknown"'::jsonb
                                    END
                                )
                            )
                            FROM jsonb_array_elements(status::jsonb->'gpu_devices') AS gpu_device
                        )::jsonb
                    )::json
                )
            """)
        )

    elif conn.dialect.name == 'sqlite':
        rows = conn.execute(sa.text("SELECT id, status FROM workers")).fetchall()
        for row in rows:
            id = row[0]
            status_content = row[1]
            status = json.loads(status_content)
            if 'gpu_devices' in status:
                for device in status.get('gpu_devices', []):
                    device['labels'] = {}
                    if device.get('vendor') == 'NVIDIA':
                        device['type'] = 'cuda'
                    elif device.get('vendor') == 'Moore Threads':
                        device['type'] = 'musa'
                    elif device.get('vendor') == 'Apple':
                        device['type'] = 'mps'
                    elif device.get('vendor') == 'Huawei':
                        device['type'] = 'npu'
                    elif device.get('vendor') == 'AMD':
                        device['type'] = 'rocm'
                    else:
                        device['type'] = 'unknown'
                conn.execute(
                    sa.text("UPDATE workers SET status = :status WHERE id = :id"),
                    {'status': json.dumps(status), 'id': id}
                )

    with op.batch_alter_table('models', schema=None) as batch_op:
        connection = batch_op.get_bind()
        categories_case = sa.case(
                (sa.column('reranker') == True, sa.literal(json.dumps(['reranker']))),
                (sa.column('embedding_only') == True, sa.literal(json.dumps(['embedding']))),
                (sa.column('image_only') == True, sa.literal(json.dumps(['image']))),
                (sa.column('speech_to_text') == True, sa.literal(json.dumps(['speech_to_text']))),
                (sa.column('text_to_speech') == True, sa.literal(json.dumps(['text_to_speech']))),
                else_=sa.literal(json.dumps(['llm']))
            )
        if connection.dialect.name == 'postgresql':
            categories_case = sa.case(
                (sa.column('reranker') == True, sa.func.to_json(['reranker'])),
                (sa.column('embedding_only') == True, sa.func.to_json(['embedding'])),
                (sa.column('image_only') == True, sa.func.to_json(['image'])),
                (sa.column('speech_to_text') == True, sa.func.to_json(['speech_to_text'])),
                (sa.column('text_to_speech') == True, sa.func.to_json(['text_to_speech'])),
                else_=sa.func.to_json(['llm'])
            )

        models_table = sa.table('models', sa.column('categories', sa.JSON))
        connection.execute(
            sa.update(models_table).values(
                categories=categories_case
            )
        )


def downgrade() -> None:
    with op.batch_alter_table('models', schema=None) as batch_op:
        batch_op.drop_column('categories')
        batch_op.drop_column('meta')

    with op.batch_alter_table('workers', schema=None) as batch_op:
        batch_op.drop_column('port')
        batch_op.drop_column('heartbeat_time')
        batch_op.drop_column('unreachable')
