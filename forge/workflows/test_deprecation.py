#!/usr/bin/env python3
"""
Test script to verify deprecation warnings for the old calculator interface.
"""

import warnings
import sys

def test_deprecation_warnings():
    """Test that deprecation warnings are shown when using the old interface."""
    
    print("Testing deprecation warnings for forge.workflows.calculator_interface...")
    
    # Capture warnings
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        
        # Test importing the deprecated module
        try:
            from forge.workflows.calculator_interface import create_calculator, check_calculator_availability
            print("✅ Successfully imported deprecated functions")
        except ImportError as e:
            print(f"❌ Import failed: {e}")
            return False
        
        # Test calling deprecated functions
        try:
            available = check_calculator_availability()
            print(f"✅ check_calculator_availability() returned: {available}")
        except Exception as e:
            print(f"❌ check_calculator_availability() failed: {e}")
        
        # Check if warnings were issued
        deprecation_warnings = [warning for warning in w if issubclass(warning.category, DeprecationWarning)]
        
        if deprecation_warnings:
            print(f"✅ {len(deprecation_warnings)} deprecation warning(s) issued:")
            for i, warning in enumerate(deprecation_warnings, 1):
                print(f"   {i}. {warning.message}")
        else:
            print("❌ No deprecation warnings were issued")
            return False
    
    print("\n✅ Deprecation warnings are working correctly!")
    return True


if __name__ == "__main__":
    success = test_deprecation_warnings()
    if success:
        print("\n🎉 Deprecation test passed!")
        sys.exit(0)
    else:
        print("\n❌ Deprecation test failed!")
        sys.exit(1) 