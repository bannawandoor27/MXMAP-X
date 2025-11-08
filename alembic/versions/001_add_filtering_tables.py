"""Add filtering tables

Revision ID: 001
Revises: 
Create Date: 2025-11-08

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '001'
down_revision = None
branch_labels = None
depends_on = None


def upgrade():
    # Create filtering_models table
    op.create_table(
        'filtering_models',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('electrolyte', sa.String(length=100), nullable=False),
        sa.Column('process', sa.String(length=100), nullable=False),
        sa.Column('Rs', sa.Float(), nullable=False),
        sa.Column('Q', sa.Float(), nullable=False),
        sa.Column('alpha', sa.Float(), nullable=False),
        sa.Column('Rleak', sa.Float(), nullable=False),
        sa.Column('calib_factors_json', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_filtering_models_id'), 'filtering_models', ['id'], unique=False)
    
    # Create filtering_runs table
    op.create_table(
        'filtering_runs',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('input_json', sa.Text(), nullable=False),
        sa.Column('kpis_json', sa.Text(), nullable=False),
        sa.Column('params_json', sa.Text(), nullable=False),
        sa.Column('area_mm2', sa.Float(), nullable=False),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_filtering_runs_id'), 'filtering_runs', ['id'], unique=False)


def downgrade():
    op.drop_index(op.f('ix_filtering_runs_id'), table_name='filtering_runs')
    op.drop_table('filtering_runs')
    op.drop_index(op.f('ix_filtering_models_id'), table_name='filtering_models')
    op.drop_table('filtering_models')
