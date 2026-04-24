import os
import pandas as pd
from pathlib import Path

data_dir = Path("data")
results = {}
total_files = 0
all_100_samples = True

print("=" * 80)
print("KIỂM TRA SỐ LƯỢNG SAMPLES CỦA CÁC FILE DỮ LIỆU")
print("=" * 80)

# Iterate through each person folder
for person_folder in sorted(data_dir.iterdir()):
    if person_folder.is_dir():
        person_name = person_folder.name
        results[person_name] = {}
        
        # Check each CSV file in the person's folder
        for csv_file in sorted(person_folder.glob("*.csv")):
            try:
                df = pd.read_csv(csv_file, header=None)
                num_samples = len(df)
                results[person_name][csv_file.name] = num_samples
                total_files += 1
                
                if num_samples != 100:
                    all_100_samples = False
            except Exception as e:
                results[person_name][csv_file.name] = f"Error: {e}"

# Display results
for person, files in results.items():
    print(f"\n📁 {person}:")
    for file_name, num_samples in sorted(files.items()):
        if isinstance(num_samples, int):
            status = "✅" if num_samples == 100 else "❌"
            print(f"   {status} {file_name}: {num_samples} samples")
        else:
            print(f"   ⚠️  {file_name}: {num_samples}")

print("\n" + "=" * 80)
print(f"Tổng số file: {total_files}")
if all_100_samples:
    print("✅ TẤT CẢ FILE ĐỀU CÓ ĐÚNG 100 SAMPLES")
else:
    print("❌ CÓ NHỮNG FILE KHÔNG CÓ ĐÚNG 100 SAMPLES")
print("=" * 80)
