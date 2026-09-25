"""shooter, and timestamps that keep their offset

Revision ID: 1601b3a8f646
Revises: 9728ec8d1ef0
Create Date: 2026-09-25 21:22:44.048857

"""

from collections.abc import Sequence

import sqlalchemy as sa
import sqlmodel
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "1601b3a8f646"
down_revision: str | None = "9728ec8d1ef0"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

TIMESTAMPS = [
    ("user", "created_at"),
    ("target_profile", "created_at"),
    ("session", "created_at"),
    ("image", "uploaded_at"),
    ("interpretation", "created_at"),
]


def _retype(table: str, column: str, old: sa.types.TypeEngine, new: sa.types.TypeEngine) -> None:
    with op.batch_alter_table(table) as batch_op:
        batch_op.alter_column(
            column,
            existing_type=old,
            type_=new,
            existing_nullable=False,
            # A naive value means UTC, both ways. Without this Postgres converts through the
            # session's time zone, and a server not set to UTC shifts every value.
            postgresql_using=f"{column} AT TIME ZONE 'UTC'",
        )


def upgrade() -> None:
    with op.batch_alter_table("session") as batch_op:
        batch_op.add_column(sa.Column("shooter", sqlmodel.sql.sqltypes.AutoString(), nullable=True))

    for table, column in TIMESTAMPS:
        _retype(table, column, sa.DateTime(), sqlmodel.sql.sqltypes.UTCDateTime())


def downgrade() -> None:
    for table, column in TIMESTAMPS:
        _retype(table, column, sqlmodel.sql.sqltypes.UTCDateTime(), sa.DateTime())

    with op.batch_alter_table("session") as batch_op:
        batch_op.drop_column("shooter")
