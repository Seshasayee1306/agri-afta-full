import pandas as pd
import numpy as np
import os

# Load existing dataset
CUR_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(CUR_DIR, "../dataset/irrigation_dataset.csv")
df = pd.read_csv(DATASET_PATH)

print(f"Original dataset size: {len(df)}")
print(f"Original class distribution:\n{df['needs_water'].value_counts()}")

# Create comprehensive augmentation data
np.random.seed(42)
n_samples = 5000  # Add 5000 new samples for better coverage

augmented_data = []

# 1. DROUGHT scenarios (needs_water = 1)
# Very dry soil + no/low rain + hot conditions
for _ in range(1500):
    row = {
        'soil_moisture': np.random.uniform(5, 20),  # Very dry
        'temperature': np.random.uniform(30, 45),   # Hot
        'soil_humidity': np.random.uniform(10, 35),
        'hour': np.random.randint(0, 24),
        'dayofyear': 1,
        'air_temp': np.random.uniform(28, 42),
        'air_humidity': np.random.uniform(20, 50),
        'rainfall': np.random.uniform(0, 10),      # Little/no rain
        'ph': np.random.uniform(6.0, 7.5),
        'nitrogen': np.random.randint(60, 120),
        'phosphorus': np.random.randint(30, 70),
        'potassium': np.random.randint(30, 70),
        'needs_water': 1.0
    }
    augmented_data.append(row)

# 2. DRY scenarios (needs_water = 1)
# Moderately dry + various conditions
for _ in range(1500):
    row = {
        'soil_moisture': np.random.uniform(20, 40),  # Moderately dry
        'temperature': np.random.uniform(25, 40),
        'soil_humidity': np.random.uniform(30, 55),
        'hour': np.random.randint(0, 24),
        'dayofyear': 1,
        'air_temp': np.random.uniform(24, 38),
        'air_humidity': np.random.uniform(30, 60),
        'rainfall': np.random.uniform(5, 50),       # Low to moderate rain
        'ph': np.random.uniform(6.0, 7.5),
        'nitrogen': np.random.randint(70, 120),
        'phosphorus': np.random.randint(35, 65),
        'potassium': np.random.randint(35, 65),
        'needs_water': 1.0
    }
    augmented_data.append(row)

# 3. WET scenarios (needs_water = 0)
# High soil moisture + rain
for _ in range(1000):
    row = {
        'soil_moisture': np.random.uniform(70, 90),  # Very wet
        'temperature': np.random.uniform(15, 30),    # Cool to moderate
        'soil_humidity': np.random.uniform(65, 95),
        'hour': np.random.randint(0, 24),
        'dayofyear': 1,
        'air_temp': np.random.uniform(18, 28),
        'air_humidity': np.random.uniform(60, 95),
        'rainfall': np.random.uniform(100, 300),     # Heavy rain
        'ph': np.random.uniform(6.0, 7.5),
        'nitrogen': np.random.randint(80, 130),
        'phosphorus': np.random.randint(40, 70),
        'potassium': np.random.randint(40, 70),
        'needs_water': 0.0
    }
    augmented_data.append(row)

# 4. MODERATE WET scenarios (needs_water = 0)
# Adequate moisture + decent rain
for _ in range(1000):
    row = {
        'soil_moisture': np.random.uniform(50, 75),  # Adequate
        'temperature': np.random.uniform(18, 32),
        'soil_humidity': np.random.uniform(50, 75),
        'hour': np.random.randint(0, 24),
        'dayofyear': 1,
        'air_temp': np.random.uniform(20, 30),
        'air_humidity': np.random.uniform(45, 75),
        'rainfall': np.random.uniform(60, 200),      # Good rain
        'ph': np.random.uniform(6.0, 7.5),
        'nitrogen': np.random.randint(75, 125),
        'phosphorus': np.random.randint(38, 68),
        'potassium': np.random.randint(38, 68),
        'needs_water': 0.0
    }
    augmented_data.append(row)

# Create augmentation dataframe
df_aug = pd.DataFrame(augmented_data)

print(f"\nAugmented data size: {len(df_aug)}")
print(f"Augmented class distribution:\n{df_aug['needs_water'].value_counts()}")

# Combine with original dataset
df_combined = pd.concat([df, df_aug], ignore_index=True)

# Shuffle the combined dataset
df_combined = df_combined.sample(frac=1, random_state=42).reset_index(drop=True)

print(f"\nCombined dataset size: {len(df_combined)}")
print(f"Combined class distribution:\n{df_combined['needs_water'].value_counts()}")

# Save augmented dataset
AUGMENTED_PATH = os.path.join(CUR_DIR, "../dataset/irrigation_dataset_augmented.csv")
df_combined.to_csv(AUGMENTED_PATH, index=False)

# Also create a backup of original
BACKUP_PATH = os.path.join(CUR_DIR, "../dataset/irrigation_dataset_original_backup.csv")
if not os.path.exists(BACKUP_PATH):
    df.to_csv(BACKUP_PATH, index=False)
    print(f"✅ Original dataset backed up to: {BACKUP_PATH}")

print(f"✅ Augmented dataset saved to: {AUGMENTED_PATH}")
print("\n🔹 Class balance improved!")
print(f"   needs_water=0: {(df_combined['needs_water']==0).sum()} ({(df_combined['needs_water']==0).sum()/len(df_combined)*100:.1f}%)")
print(f"   needs_water=1: {(df_combined['needs_water']==1).sum()} ({(df_combined['needs_water']==1).sum()/len(df_combined)*100:.1f}%)")
