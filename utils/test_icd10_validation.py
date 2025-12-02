#!/usr/bin/env python3
"""
Quick test to verify ICD-10 code validation is working correctly
"""

import icd10 as cm

def is_valid_icd10_code(code):
    """
    Check if a code exists in the ICD-10 knowledge graph
    Returns True if valid, False otherwise
    """
    if not code or not isinstance(code, str):
        return False
    
    try:
        # First try the built-in validation
        is_valid = cm.is_valid_item(code)
        if not is_valid:
            return False
        
        # Double-check by trying to get the data
        _ = cm.get_full_data(code, search_in_ancestors=False, prioritize_blocks=False)
        return True
    except Exception as e:
        return False

# Test with known valid and invalid codes
test_codes = [
    # Valid ICD-10 codes
    ("S62512B", True, "Valid - Fracture code with 7th char"),
    ("K219", True, "Valid - GERD"),
    ("F329", True, "Valid - Depression"),
    ("S52", True, "Valid - Fracture of forearm"),
    
    # Invalid codes (procedure codes, not ICD-10 diagnosis)
    ("0LB80ZZ", False, "Invalid - ICD-10-PCS procedure code"),
    ("0PSV04Z", False, "Invalid - ICD-10-PCS procedure code"),
    ("0LQ80ZZ", False, "Invalid - ICD-10-PCS procedure code"),
    ("0HQGXZZ", False, "Invalid - ICD-10-PCS procedure code"),
]

print("="*80)
print("ICD-10 Code Validation Test")
print("="*80)
print()

passed = 0
failed = 0

for code, expected_valid, description in test_codes:
    result = is_valid_icd10_code(code)
    status = "✅ PASS" if result == expected_valid else "❌ FAIL"
    
    if result == expected_valid:
        passed += 1
    else:
        failed += 1
    
    print(f"{status} | {code:12} | Expected: {str(expected_valid):5} | Got: {str(result):5} | {description}")

print()
print("="*80)
print(f"Results: {passed} passed, {failed} failed out of {len(test_codes)} tests")
print("="*80)

if failed == 0:
    print("✅ All tests passed! Validation is working correctly.")
else:
    print("⚠️  Some tests failed. Check the validation logic.")
