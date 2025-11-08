"""Seed database with synthetic training data."""

import asyncio
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from sqlalchemy import select
from app.db.session import async_session_maker, engine, Base
from app.models.database import Device, TrainingMetadata
from datetime import datetime


async def create_tables() -> None:
    """Create all database tables."""
    print("Creating database tables...")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    print("✓ Tables created")


async def seed_devices(csv_path: str = "data/synthetic_training_data.csv") -> None:
    """
    Seed database with device data from CSV.
    
    Args:
        csv_path: Path to synthetic data CSV file
    """
    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)
    print(f"✓ Loaded {len(df)} devices")
    
    async with async_session_maker() as session:
        # Check if data already exists
        result = await session.execute(select(Device))
        existing_count = len(result.scalars().all())
        
        if existing_count > 0:
            print(f"⚠ Database already contains {existing_count} devices")
            response = input("Clear existing data and reseed? (y/N): ")
            if response.lower() != 'y':
                print("Seeding cancelled")
                return
            
            # Clear existing data
            await session.execute(Device.__table__.delete())
            await session.commit()
            print("✓ Cleared existing devices")
        
        # Insert devices
        print("Inserting devices...")
        devices = []
        for _, row in df.iterrows():
            device = Device(
                mxene_type=row["mxene_type"],
                terminations=row["terminations"],
                electrolyte=row["electrolyte"],
                electrolyte_concentration=row["electrolyte_concentration"] if pd.notna(row["electrolyte_concentration"]) else None,
                thickness_um=row["thickness_um"],
                deposition_method=row["deposition_method"],
                annealing_temp_c=row["annealing_temp_c"] if pd.notna(row["annealing_temp_c"]) else None,
                annealing_time_min=row["annealing_time_min"] if pd.notna(row["annealing_time_min"]) else None,
                interlayer_spacing_nm=row["interlayer_spacing_nm"] if pd.notna(row["interlayer_spacing_nm"]) else None,
                specific_surface_area_m2g=row["specific_surface_area_m2g"] if pd.notna(row["specific_surface_area_m2g"]) else None,
                pore_volume_cm3g=row["pore_volume_cm3g"] if pd.notna(row["pore_volume_cm3g"]) else None,
                optical_transmittance=row["optical_transmittance"] if pd.notna(row["optical_transmittance"]) else None,
                sheet_resistance_ohm_sq=row["sheet_resistance_ohm_sq"] if pd.notna(row["sheet_resistance_ohm_sq"]) else None,
                areal_capacitance_mf_cm2=row["areal_capacitance_mf_cm2"],
                esr_ohm=row["esr_ohm"],
                rate_capability_percent=row["rate_capability_percent"],
                cycle_life_cycles=int(row["cycle_life_cycles"]),
                source=row["source"],
                notes=row["notes"],
            )
            devices.append(device)
        
        session.add_all(devices)
        await session.commit()
        print(f"✓ Inserted {len(devices)} devices")


async def seed_model_metadata() -> None:
    """Seed initial model metadata."""
    print("Creating model metadata...")
    
    async with async_session_maker() as session:
        # Check if metadata exists
        result = await session.execute(select(TrainingMetadata))
        existing = result.scalar_one_or_none()
        
        if existing:
            print("⚠ Model metadata already exists")
            return
        
        # Create dummy model metadata
        metadata = TrainingMetadata(
            model_version="v0.1.0-dummy",
            model_type="dummy",
            train_r2_capacitance=0.95,
            train_r2_esr=0.88,
            train_r2_rate_capability=0.82,
            train_r2_cycle_life=0.79,
            test_r2_capacitance=0.93,
            test_r2_esr=0.85,
            test_r2_rate_capability=0.80,
            test_r2_cycle_life=0.76,
            train_rmse_capacitance=25.5,
            test_rmse_capacitance=28.3,
            training_samples=240,
            test_samples=60,
            hyperparameters='{"type": "dummy", "seed": 42}',
            trained_at=datetime.utcnow(),
            is_active=1,
        )
        
        session.add(metadata)
        await session.commit()
        print("✓ Created model metadata")


async def main() -> None:
    """Main seeding function."""
    print("=" * 60)
    print("MXMAP-X Database Seeding")
    print("=" * 60)
    
    try:
        # Create tables
        await create_tables()
        
        # Seed devices
        await seed_devices()
        
        # Seed model metadata
        await seed_model_metadata()
        
        print("\n" + "=" * 60)
        print("✓ Database seeding completed successfully!")
        print("=" * 60)
        
        # Print summary
        async with async_session_maker() as session:
            device_count = await session.execute(select(func.count(Device.id)))
            total_devices = device_count.scalar_one()
            
            print(f"\nDatabase summary:")
            print(f"  - Total devices: {total_devices}")
            print(f"  - Active model: v0.1.0-dummy")
            print(f"\nYou can now start the API server:")
            print(f"  uvicorn app.main:app --reload")
        
    except Exception as e:
        print(f"\n✗ Error during seeding: {str(e)}")
        raise
    finally:
        await engine.dispose()


if __name__ == "__main__":
    # Import func here to avoid circular imports
    from sqlalchemy import func
    asyncio.run(main())
