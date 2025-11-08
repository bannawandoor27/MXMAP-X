"""Add printing tables

Revision ID: 002
Revises: 001
Create Date: 2025-11-08

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '002'
down_revision = '001'
branch_labels = None
depends_on = None


def upgrade():
    # Create printing_calibrations table
    op.create_table(
        'printing_calibrations',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('process', sa.String(length=50), nullable=False),
        sa.Column('version', sa.String(length=50), nullable=False),
        sa.Column('coeffs_json', sa.Text(), nullable=False),
        sa.Column('metrics_json', sa.Text(), nullable=False),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_printing_calibrations_id'), 'printing_calibrations', ['id'], unique=False)
    
    # Create printing_runs table
    op.create_table(
        'printing_runs',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('input_json', sa.Text(), nullable=False),
        sa.Column('output_json', sa.Text(), nullable=False),
        sa.Column('manufacturability_score', sa.Integer(), nullable=False),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_printing_runs_id'), 'printing_runs', ['id'], unique=False)


def downgrade():
    op.drop_index(op.f('ix_printing_runs_id'), table_name='printing_runs')
    op.drop_table('printing_runs')
    op.drop_index(op.f('ix_printing_calibrations_id'), table_name='printing_calibrations')
    op.drop_table('printing_calibrations')
