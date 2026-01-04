"""Initial schema - all tables

Revision ID: 001_initial
Revises:
Create Date: 2026-01-03

Creates all tables:
- user_mappings
- conversations
- messages
- tool_executions
- audit_logs
- hallucination_reports

IMPORTANT: This migration runs with admin_user privileges.
After tables are created, permissions are set for bot_user.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers
revision: str = '001_initial'
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # === USER MAPPINGS ===
    op.create_table(
        'user_mappings',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('phone_number', sa.String(20), nullable=False, unique=True),
        sa.Column('api_identity', sa.String(100), nullable=False),
        sa.Column('display_name', sa.String(200), nullable=True),
        sa.Column('tenant_id', sa.String(100), nullable=True),
        sa.Column('is_active', sa.Boolean(), default=True),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.Column('updated_at', sa.DateTime(), nullable=True),
    )
    op.create_index('ix_user_phone', 'user_mappings', ['phone_number'])
    op.create_index('ix_user_tenant', 'user_mappings', ['tenant_id'])

    # === CONVERSATIONS ===
    op.create_table(
        'conversations',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('user_id', postgresql.UUID(as_uuid=True),
                  sa.ForeignKey('user_mappings.id', ondelete='CASCADE'), nullable=False),
        sa.Column('started_at', sa.DateTime(), nullable=True),
        sa.Column('last_activity', sa.DateTime(), nullable=True),
        sa.Column('message_count', sa.Integer(), default=0),
        sa.Column('state', sa.String(50), default='active'),
    )
    op.create_index('ix_conv_user', 'conversations', ['user_id'])
    op.create_index('ix_conv_activity', 'conversations', ['last_activity'])

    # === MESSAGES ===
    op.create_table(
        'messages',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('conversation_id', postgresql.UUID(as_uuid=True),
                  sa.ForeignKey('conversations.id', ondelete='CASCADE'), nullable=False),
        sa.Column('role', sa.String(20), nullable=False),
        sa.Column('content', sa.Text(), nullable=False),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.Column('metadata', postgresql.JSON(), nullable=True),
    )
    op.create_index('ix_msg_conv', 'messages', ['conversation_id'])
    op.create_index('ix_msg_created', 'messages', ['created_at'])

    # === TOOL EXECUTIONS ===
    op.create_table(
        'tool_executions',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('conversation_id', postgresql.UUID(as_uuid=True),
                  sa.ForeignKey('conversations.id', ondelete='SET NULL'), nullable=True),
        sa.Column('tool_name', sa.String(200), nullable=False),
        sa.Column('parameters', postgresql.JSON(), nullable=True),
        sa.Column('result', postgresql.JSON(), nullable=True),
        sa.Column('success', sa.Boolean(), nullable=True),
        sa.Column('error_message', sa.Text(), nullable=True),
        sa.Column('execution_time_ms', sa.Integer(), nullable=True),
        sa.Column('executed_at', sa.DateTime(), nullable=True),
    )
    op.create_index('ix_tool_name', 'tool_executions', ['tool_name'])
    op.create_index('ix_tool_executed', 'tool_executions', ['executed_at'])

    # === AUDIT LOGS (admin only!) ===
    op.create_table(
        'audit_logs',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('user_id', sa.String(100), nullable=True),
        sa.Column('action', sa.String(100), nullable=False),
        sa.Column('entity_type', sa.String(100), nullable=True),
        sa.Column('entity_id', sa.String(100), nullable=True),
        sa.Column('details', postgresql.JSON(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=True),
    )
    op.create_index('ix_audit_action', 'audit_logs', ['action'])
    op.create_index('ix_audit_created', 'audit_logs', ['created_at'])
    op.create_index('ix_audit_entity', 'audit_logs', ['entity_type', 'entity_id'])

    # === HALLUCINATION REPORTS (admin read, bot insert only!) ===
    op.create_table(
        'hallucination_reports',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('user_query', sa.Text(), nullable=False),
        sa.Column('bot_response', sa.Text(), nullable=False),
        sa.Column('user_feedback', sa.Text(), nullable=False),
        sa.Column('conversation_id', sa.String(50), nullable=True),
        sa.Column('tenant_id', sa.String(100), nullable=True),
        sa.Column('model', sa.String(50), nullable=True),
        sa.Column('retrieved_chunks', postgresql.JSON(), nullable=True),
        sa.Column('api_raw_response', postgresql.JSON(), nullable=True),
        sa.Column('reviewed', sa.Boolean(), default=False),
        sa.Column('reviewed_by', sa.String(100), nullable=True),
        sa.Column('reviewed_at', sa.DateTime(), nullable=True),
        sa.Column('correction', sa.Text(), nullable=True),
        sa.Column('category', sa.String(50), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=True),
    )
    op.create_index('ix_halluc_unreviewed', 'hallucination_reports', ['reviewed', 'created_at'])
    op.create_index('ix_halluc_tenant', 'hallucination_reports', ['tenant_id', 'category'])
    op.create_index('ix_halluc_conv', 'hallucination_reports', ['conversation_id'])

    # === SET PERMISSIONS ===
    # Bot user: limited access
    op.execute("GRANT SELECT, INSERT, UPDATE, DELETE ON user_mappings TO bot_user")
    op.execute("GRANT SELECT, INSERT, UPDATE, DELETE ON conversations TO bot_user")
    op.execute("GRANT SELECT, INSERT, UPDATE, DELETE ON messages TO bot_user")
    op.execute("GRANT SELECT, INSERT, UPDATE, DELETE ON tool_executions TO bot_user")
    op.execute("GRANT INSERT ON hallucination_reports TO bot_user")  # INSERT ONLY!
    op.execute("GRANT USAGE ON ALL SEQUENCES IN SCHEMA public TO bot_user")

    # Admin user: full access
    op.execute("GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO admin_user")
    op.execute("GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO admin_user")

    # Revoke admin tables from bot
    op.execute("REVOKE ALL ON audit_logs FROM bot_user")
    op.execute("REVOKE SELECT, UPDATE, DELETE ON hallucination_reports FROM bot_user")


def downgrade() -> None:
    op.drop_table('hallucination_reports')
    op.drop_table('audit_logs')
    op.drop_table('tool_executions')
    op.drop_table('messages')
    op.drop_table('conversations')
    op.drop_table('user_mappings')
