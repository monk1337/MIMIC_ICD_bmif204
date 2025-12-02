#!/usr/bin/env python3
"""
Check ICD-10 module data loading
"""

import icd10

print("="*80)
print("ICD-10 MODULE DIAGNOSTIC")
print("="*80)
print()

print(f"1. code_to_node dictionary size: {len(icd10.code_to_node)}")
print(f"2. chapter_list size: {len(icd10.chapter_list)}")
print(f"3. all_codes_list size: {len(icd10.all_codes_list)}")
print(f"4. all_codes_list_no_dots size: {len(icd10.all_codes_list_no_dots)}")

print()
print("Chapters:")
for i, chapter in enumerate(icd10.chapter_list[:5], 1):
    print(f"  {i}. {chapter.name} - {chapter.description[:50]}...")

print()
print("Sample codes from code_to_node:")
sample_codes = list(icd10.code_to_node.keys())[:10]
for code in sample_codes:
    node = icd10.code_to_node[code]
    print(f"  {code} - {node.description[:50]}...")

print()
print("Testing get_full_data on known codes:")
test_codes = ["I50", "I4891", "E119", "I2101"]
for code in test_codes:
    try:
        data = icd10.get_full_data(code)
        if data:
            print(f"  ✅ {code}: {data['description'][:50]}...")
        else:
            print(f"  ❌ {code}: No data returned")
    except Exception as e:
        print(f"  ❌ {code}: Error - {str(e)}")

print()
print("="*80)
