"""add secrets tables

Create Date: 2025-01-27 00:00:00.000000

"""

import time

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision = "1b49d398cd23"
down_revision = "bf29a5ff90ea"
branch_labels = None
depends_on = None


def upgrade():
    # Create secrets table (API keys)
    op.create_table(
        "secrets",
        sa.Column("secret_id", sa.String(length=36), nullable=False),
        sa.Column("secret_name", sa.String(length=255), nullable=False),
        sa.Column("encrypted_value", sa.LargeBinary(), nullable=False),
        sa.Column("wrapped_dek", sa.LargeBinary(), nullable=False),
        sa.Column("kek_version", sa.Integer(), nullable=False, default=1),
        sa.Column("masked_value", sa.String(length=100), nullable=False),
        sa.Column("provider", sa.String(length=64), nullable=True),
        # auth_config: encrypted JSON for complex provider authentication (AWS, Azure, etc.)
        sa.Column("encrypted_auth_config", sa.LargeBinary(), nullable=True),
        sa.Column("wrapped_auth_config_dek", sa.LargeBinary(), nullable=True),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("is_shared", sa.Boolean(), nullable=False, default=False),
        sa.Column("created_by", sa.String(length=255), nullable=True),
        sa.Column(
            "created_at",
            sa.BigInteger(),
            default=lambda: int(time.time() * 1000),
            nullable=False,
        ),
        sa.Column("last_updated_by", sa.String(length=255), nullable=True),
        sa.Column(
            "last_updated_at",
            sa.BigInteger(),
            default=lambda: int(time.time() * 1000),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("secret_id", name="secrets_pk"),
    )
    with op.batch_alter_table("secrets", schema=None) as batch_op:
        batch_op.create_index(
            "index_secrets_is_shared_secret_name", ["is_shared", "secret_name"], unique=False
        )

    # Create secret_tags table
    op.create_table(
        "secret_tags",
        sa.Column("secret_id", sa.String(length=36), nullable=False),
        sa.Column("key", sa.String(length=250), nullable=False),
        sa.Column("value", sa.String(length=5000), nullable=True),
        sa.ForeignKeyConstraint(
            ["secret_id"],
            ["secrets.secret_id"],
            name="fk_secret_tags_secret_id",
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("secret_id", "key", name="secret_tags_pk"),
    )
    with op.batch_alter_table("secret_tags", schema=None) as batch_op:
        batch_op.create_index("index_secret_tags_secret_id", ["secret_id"], unique=False)

    # Create routes table (model configurations)
    op.create_table(
        "routes",
        sa.Column("route_id", sa.String(length=36), nullable=False),
        sa.Column("secret_id", sa.String(length=36), nullable=False),
        sa.Column("model_name", sa.String(length=256), nullable=False),
        sa.Column("name", sa.String(length=255), nullable=True),  # Optional display name
        sa.Column("description", sa.Text(), nullable=True),
        # model_config: encrypted JSON for runtime parameters (temperature, max_tokens, etc.)
        sa.Column("encrypted_model_config", sa.LargeBinary(), nullable=True),
        sa.Column("wrapped_model_config_dek", sa.LargeBinary(), nullable=True),
        sa.Column("created_by", sa.String(length=255), nullable=True),
        sa.Column(
            "created_at",
            sa.BigInteger(),
            default=lambda: int(time.time() * 1000),
            nullable=False,
        ),
        sa.Column("last_updated_by", sa.String(length=255), nullable=True),
        sa.Column(
            "last_updated_at",
            sa.BigInteger(),
            default=lambda: int(time.time() * 1000),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["secret_id"],
            ["secrets.secret_id"],
            name="fk_routes_secret_id",
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("route_id", name="routes_pk"),
        sa.UniqueConstraint("name", name="unique_route_name"),
    )
    with op.batch_alter_table("routes", schema=None) as batch_op:
        batch_op.create_index("index_routes_secret_id", ["secret_id"], unique=False)
        batch_op.create_index("index_routes_model_name", ["model_name"], unique=False)

    # Create route_tags table
    op.create_table(
        "route_tags",
        sa.Column("route_id", sa.String(length=36), nullable=False),
        sa.Column("key", sa.String(length=250), nullable=False),
        sa.Column("value", sa.String(length=5000), nullable=True),
        sa.ForeignKeyConstraint(
            ["route_id"],
            ["routes.route_id"],
            name="fk_route_tags_route_id",
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("route_id", "key", name="route_tags_pk"),
    )
    with op.batch_alter_table("route_tags", schema=None) as batch_op:
        batch_op.create_index("index_route_tags_route_id", ["route_id"], unique=False)

    # Create bindings table (route-to-resource associations)
    op.create_table(
        "secrets_bindings",
        sa.Column("binding_id", sa.String(length=36), nullable=False),
        sa.Column("route_id", sa.String(length=36), nullable=False),
        sa.Column("resource_type", sa.String(length=50), nullable=False),
        sa.Column("resource_id", sa.String(length=255), nullable=False),
        sa.Column("field_name", sa.String(length=255), nullable=False),
        sa.Column(
            "created_at",
            sa.BigInteger(),
            default=lambda: int(time.time() * 1000),
            nullable=False,
        ),
        sa.Column("created_by", sa.String(length=255), nullable=True),
        sa.Column(
            "last_updated_at",
            sa.BigInteger(),
            default=lambda: int(time.time() * 1000),
            nullable=False,
        ),
        sa.Column("last_updated_by", sa.String(length=255), nullable=True),
        sa.ForeignKeyConstraint(
            ["route_id"],
            ["routes.route_id"],
            name="fk_secrets_bindings_route_id",
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("binding_id", name="secrets_bindings_pk"),
    )
    with op.batch_alter_table("secrets_bindings", schema=None) as batch_op:
        batch_op.create_index("index_secrets_bindings_route_id", ["route_id"], unique=False)
        batch_op.create_index(
            "index_secrets_bindings_resource_type_resource_id",
            ["resource_type", "resource_id"],
            unique=False,
        )


def downgrade():
    op.drop_table("secrets_bindings")
    op.drop_table("route_tags")
    op.drop_table("routes")
    op.drop_table("secret_tags")
    op.drop_table("secrets")
