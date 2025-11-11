"""add the mapping of model backend field

Revision ID: bb9ce21dcc10
Revises: 89cb8df41bf0
Create Date: 2025-11-12 11:57:57.230401

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
import sqlmodel
import gpustack
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = 'bb9ce21dcc10'
down_revision: Union[str, None] = '89cb8df41bf0'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("""
UPDATE models
SET backend = CASE backend
    WHEN 'vllm' THEN 'vLLM'
    WHEN 'ascend-mindie' THEN 'MindIE'
    WHEN 'vox-box' THEN 'VoxBox'
    ELSE backend
END;
""")


def downgrade() -> None:
    op.execute("""
UPDATE models
SET backend = CASE backend
    WHEN 'vLLM' THEN 'vllm'
    WHEN 'MindIE' THEN 'ascend-mindie'
    WHEN 'VoxBox' THEN 'vox-box'
    ELSE backend
END;
""")
