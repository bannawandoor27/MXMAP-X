"""Generate physics-informed synthetic training data for MXMAP-X."""

import numpy as np
import pandas as pd
from typing import Any


class SyntheticDataGenerator:
    """
    Generate realistic MXene supercapacitor training data.
    
    Uses physics-informed correlations and realistic noise to create
    300 synthetic device samples for model training.
    """

    def __init__(self, n_samples: int = 300, seed: int = 42) -> None:
        """
        Initialize generator.
        
        Args:
            n_samples: Number of samples to generate
            seed: Random seed for reproducibility
        """
        self.n_samples = n_samples
        self.rng = np.random.default_rng(seed)

    def generate(self) -> pd.DataFrame:
        """
        Generate complete synthetic dataset.
        
        Returns:
            DataFrame with device compositions and performance metrics
        """
        data = []

        for _ in range(self.n_samples):
            device = self._generate_device()
            data.append(device)

        df = pd.DataFrame(data)
        
        # Add missing values to optical properties (10-15%)
        missing_mask = self.rng.random(self.n_samples) < 0.12
        df.loc[missing_mask, "optical_transmittance"] = None
        
        missing_mask = self.rng.random(self.n_samples) < 0.15
        df.loc[missing_mask, "sheet_resistance_ohm_sq"] = None

        return df

    def _generate_device(self) -> dict[str, Any]:
        """Generate a single device with correlated properties."""
        # Material composition
        mxene_type = self.rng.choice([
            "Ti3C2Tx", "Ti3C2Tx", "Ti3C2Tx",  # Ti3C2Tx most common
            "Mo2CTx", "V2CTx", "Ti2CTx", "Nb2CTx"
        ])
        
        terminations = self.rng.choice(["O", "OH", "F", "mixed"], p=[0.4, 0.3, 0.2, 0.1])
        
        electrolyte = self.rng.choice([
            "H2SO4", "KOH", "NaOH", "ionic_liquid",
            "EMIMBF4", "PVA_H2SO4", "PVA_KOH"
        ], p=[0.25, 0.20, 0.10, 0.15, 0.10, 0.12, 0.08])
        
        electrolyte_concentration = None
        if electrolyte in ["H2SO4", "KOH", "NaOH"]:
            electrolyte_concentration = self.rng.uniform(0.5, 3.0)
        
        # Processing parameters
        thickness_um = self.rng.lognormal(mean=np.log(5.0), sigma=0.6)
        thickness_um = np.clip(thickness_um, 0.5, 40.0)
        
        deposition_method = self.rng.choice([
            "vacuum_filtration", "spray_coating", "drop_casting",
            "spin_coating", "blade_coating"
        ], p=[0.35, 0.25, 0.20, 0.12, 0.08])
        
        # Annealing (50% of samples)
        annealing_temp_c = None
        annealing_time_min = None
        if self.rng.random() < 0.5:
            annealing_temp_c = self.rng.uniform(80, 250)
            annealing_time_min = self.rng.uniform(30, 180)
        
        # Structural properties
        interlayer_spacing_nm = self.rng.normal(1.2, 0.2)
        interlayer_spacing_nm = np.clip(interlayer_spacing_nm, 0.8, 2.0)
        
        specific_surface_area_m2g = self.rng.normal(80, 25)
        specific_surface_area_m2g = np.clip(specific_surface_area_m2g, 20, 200)
        
        pore_volume_cm3g = self.rng.normal(0.12, 0.05)
        pore_volume_cm3g = np.clip(pore_volume_cm3g, 0.02, 0.5)
        
        # Optical properties (will add missing values later)
        optical_transmittance = self.rng.uniform(40, 95)
        sheet_resistance_ohm_sq = self.rng.lognormal(mean=np.log(50), sigma=0.8)
        sheet_resistance_ohm_sq = np.clip(sheet_resistance_ohm_sq, 5, 500)
        
        # Calculate performance metrics with physics-informed correlations
        capacitance = self._calculate_capacitance(
            mxene_type, electrolyte, thickness_um,
            specific_surface_area_m2g, annealing_temp_c
        )
        
        esr = self._calculate_esr(
            thickness_um, electrolyte, sheet_resistance_ohm_sq
        )
        
        rate_capability = self._calculate_rate_capability(
            thickness_um, interlayer_spacing_nm, pore_volume_cm3g,
            deposition_method
        )
        
        cycle_life = self._calculate_cycle_life(
            mxene_type, electrolyte, annealing_temp_c
        )
        
        return {
            # Material composition
            "mxene_type": mxene_type,
            "terminations": terminations,
            "electrolyte": electrolyte,
            "electrolyte_concentration": electrolyte_concentration,
            "thickness_um": round(thickness_um, 2),
            "deposition_method": deposition_method,
            "annealing_temp_c": round(annealing_temp_c, 1) if annealing_temp_c else None,
            "annealing_time_min": round(annealing_time_min, 1) if annealing_time_min else None,
            # Structural properties
            "interlayer_spacing_nm": round(interlayer_spacing_nm, 3),
            "specific_surface_area_m2g": round(specific_surface_area_m2g, 1),
            "pore_volume_cm3g": round(pore_volume_cm3g, 3),
            # Optical properties
            "optical_transmittance": round(optical_transmittance, 1),
            "sheet_resistance_ohm_sq": round(sheet_resistance_ohm_sq, 1),
            # Performance metrics
            "areal_capacitance_mf_cm2": round(capacitance, 2),
            "esr_ohm": round(esr, 3),
            "rate_capability_percent": round(rate_capability, 2),
            "cycle_life_cycles": int(cycle_life),
            # Metadata
            "source": "synthetic_data",
            "notes": "Physics-informed synthetic data for model training",
        }

    def _calculate_capacitance(
        self,
        mxene_type: str,
        electrolyte: str,
        thickness_um: float,
        surface_area: float,
        annealing_temp: float | None,
    ) -> float:
        """Calculate areal capacitance with physics-informed correlations."""
        # Base capacitance by MXene type
        base_cap = {
            "Ti3C2Tx": 300.0,
            "Mo2CTx": 250.0,
            "V2CTx": 280.0,
            "Ti2CTx": 220.0,
            "Nb2CTx": 200.0,
        }
        capacitance = base_cap.get(mxene_type, 250.0)
        
        # Thickness effect (thicker → higher capacitance)
        capacitance += thickness_um * 12.0
        
        # Electrolyte effect
        electrolyte_factor = {
            "H2SO4": 1.25,
            "KOH": 1.15,
            "NaOH": 1.12,
            "ionic_liquid": 1.08,
            "EMIMBF4": 1.08,
            "PVA_H2SO4": 1.20,
            "PVA_KOH": 1.12,
        }
        capacitance *= electrolyte_factor.get(electrolyte, 1.0)
        
        # Surface area effect
        capacitance += surface_area * 0.4
        
        # Annealing effect
        if annealing_temp:
            if 100 <= annealing_temp <= 200:
                capacitance *= 1.12
            elif annealing_temp > 250:
                capacitance *= 0.92
        
        # Add realistic noise (±10%)
        noise = self.rng.normal(0, capacitance * 0.10)
        capacitance = max(50.0, capacitance + noise)
        
        return capacitance

    def _calculate_esr(
        self,
        thickness_um: float,
        electrolyte: str,
        sheet_resistance: float,
    ) -> float:
        """Calculate ESR with correlations."""
        # Base ESR
        esr = 1.5
        
        # Thickness effect
        esr += thickness_um * 0.06
        
        # Electrolyte effect
        if electrolyte in ["ionic_liquid", "EMIMBF4"]:
            esr *= 1.6
        elif electrolyte in ["H2SO4", "KOH"]:
            esr *= 0.75
        
        # Sheet resistance correlation
        esr += sheet_resistance * 0.008
        
        # Add noise (±12%)
        noise = self.rng.normal(0, esr * 0.12)
        esr = max(0.2, esr + noise)
        
        return esr

    def _calculate_rate_capability(
        self,
        thickness_um: float,
        interlayer_spacing: float,
        pore_volume: float,
        deposition_method: str,
    ) -> float:
        """Calculate rate capability with correlations."""
        # Base rate capability
        rate_cap = 85.0
        
        # Thickness effect (thicker → worse)
        rate_cap -= thickness_um * 0.7
        
        # Interlayer spacing effect
        if interlayer_spacing > 1.4:
            rate_cap += 6.0
        elif interlayer_spacing < 1.0:
            rate_cap -= 4.0
        
        # Pore volume effect
        rate_cap += pore_volume * 12.0
        
        # Deposition method effect
        method_bonus = {
            "spray_coating": 4.0,
            "vacuum_filtration": 2.0,
            "blade_coating": 1.0,
            "spin_coating": 0.0,
            "drop_casting": -3.0,
        }
        rate_cap += method_bonus.get(deposition_method, 0.0)
        
        # Add noise
        noise = self.rng.normal(0, 4.0)
        rate_cap = np.clip(rate_cap + noise, 35.0, 98.0)
        
        return rate_cap

    def _calculate_cycle_life(
        self,
        mxene_type: str,
        electrolyte: str,
        annealing_temp: float | None,
    ) -> float:
        """Calculate cycle life with correlations."""
        # Base cycle life
        cycle_life = 10000.0
        
        # Electrolyte effect
        if electrolyte == "H2SO4":
            cycle_life *= 0.65
        elif electrolyte in ["ionic_liquid", "EMIMBF4"]:
            cycle_life *= 1.35
        elif electrolyte in ["PVA_H2SO4", "PVA_KOH"]:
            cycle_life *= 1.55
        
        # Annealing effect
        if annealing_temp and annealing_temp > 100:
            cycle_life *= 1.25
        
        # MXene type effect
        if mxene_type == "Ti3C2Tx":
            cycle_life *= 1.15
        elif mxene_type == "V2CTx":
            cycle_life *= 0.85
        
        # Add noise (±15%)
        noise = self.rng.normal(0, cycle_life * 0.15)
        cycle_life = max(1000.0, cycle_life + noise)
        
        return cycle_life


def main() -> None:
    """Generate and save synthetic data."""
    print("Generating synthetic training data...")
    
    generator = SyntheticDataGenerator(n_samples=300, seed=42)
    df = generator.generate()
    
    # Save to CSV
    output_path = "data/synthetic_training_data.csv"
    df.to_csv(output_path, index=False)
    
    print(f"✓ Generated {len(df)} samples")
    print(f"✓ Saved to {output_path}")
    print(f"\nDataset statistics:")
    print(f"  - MXene types: {df['mxene_type'].nunique()}")
    print(f"  - Electrolytes: {df['electrolyte'].nunique()}")
    print(f"  - Thickness range: {df['thickness_um'].min():.1f} - {df['thickness_um'].max():.1f} μm")
    print(f"  - Capacitance range: {df['areal_capacitance_mf_cm2'].min():.1f} - {df['areal_capacitance_mf_cm2'].max():.1f} mF/cm²")
    print(f"  - Missing optical transmittance: {df['optical_transmittance'].isna().sum()} ({df['optical_transmittance'].isna().sum()/len(df)*100:.1f}%)")
    print(f"  - Missing sheet resistance: {df['sheet_resistance_ohm_sq'].isna().sum()} ({df['sheet_resistance_ohm_sq'].isna().sum()/len(df)*100:.1f}%)")


if __name__ == "__main__":
    import os
    os.makedirs("data", exist_ok=True)
    main()
