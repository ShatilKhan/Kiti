"""
Validation script to check project structure and Python syntax.
This script does not require external dependencies to be installed.
"""
import ast
import os
from pathlib import Path


def validate_python_syntax(file_path):
    """Validate that a Python file has correct syntax."""
    try:
        with open(file_path, 'r') as f:
            ast.parse(f.read())
        return True, None
    except SyntaxError as e:
        return False, str(e)


def check_project_structure():
    """Check that the project has the expected structure."""
    project_root = Path(__file__).parent
    
    expected_structure = {
        'directories': [
            'src',
            'config',
            'data',
            'output'
        ],
        'files': [
            'src/__init__.py',
            'src/main.py',
            'src/video_processor.py',
            'src/object_detection.py',
            'src/optical_flow.py',
            'config/__init__.py',
            'config/settings.py',
            'requirements.txt',
            'setup.py',
            'README.md',
            'run.py',
            'data/README.md',
            'output/README.md'
        ]
    }
    
    print("=" * 60)
    print("Project Structure Validation")
    print("=" * 60)
    
    all_passed = True
    
    # Check directories
    print("\nChecking directories...")
    for directory in expected_structure['directories']:
        dir_path = project_root / directory
        if dir_path.exists() and dir_path.is_dir():
            print(f"  ✓ {directory}")
        else:
            print(f"  ✗ {directory} - MISSING")
            all_passed = False
    
    # Check files
    print("\nChecking files...")
    for file in expected_structure['files']:
        file_path = project_root / file
        if file_path.exists() and file_path.is_file():
            print(f"  ✓ {file}")
        else:
            print(f"  ✗ {file} - MISSING")
            all_passed = False
    
    return all_passed


def validate_python_files():
    """Validate syntax of all Python files."""
    project_root = Path(__file__).parent
    
    python_files = [
        'src/__init__.py',
        'src/main.py',
        'src/video_processor.py',
        'src/object_detection.py',
        'src/optical_flow.py',
        'config/__init__.py',
        'config/settings.py',
        'setup.py',
        'run.py'
    ]
    
    print("\n" + "=" * 60)
    print("Python Syntax Validation")
    print("=" * 60 + "\n")
    
    all_passed = True
    
    for file_path in python_files:
        full_path = project_root / file_path
        if full_path.exists():
            is_valid, error = validate_python_syntax(full_path)
            if is_valid:
                print(f"  ✓ {file_path}")
            else:
                print(f"  ✗ {file_path} - SYNTAX ERROR: {error}")
                all_passed = False
        else:
            print(f"  ✗ {file_path} - FILE NOT FOUND")
            all_passed = False
    
    return all_passed


def check_requirements():
    """Check requirements.txt format."""
    project_root = Path(__file__).parent
    req_file = project_root / 'requirements.txt'
    
    print("\n" + "=" * 60)
    print("Requirements File Validation")
    print("=" * 60 + "\n")
    
    if not req_file.exists():
        print("  ✗ requirements.txt not found")
        return False
    
    with open(req_file, 'r') as f:
        lines = f.readlines()
    
    required_packages = [
        'opencv-python',
        'ultralytics',
        'numpy',
        'matplotlib',
        'torch',
        'torchvision',
        'pillow',
        'pyyaml',
        'scikit-learn'
    ]
    
    found_packages = set()
    for line in lines:
        line = line.strip()
        if line and not line.startswith('#'):
            package_name = line.split('>=')[0].split('==')[0].strip()
            found_packages.add(package_name)
    
    all_found = True
    for package in required_packages:
        if package in found_packages:
            print(f"  ✓ {package}")
        else:
            print(f"  ✗ {package} - MISSING")
            all_found = False
    
    return all_found


def main():
    """Run all validations."""
    print("\n" + "=" * 70)
    print("  OPTICAL FLOW PROJECT - VALIDATION REPORT")
    print("=" * 70 + "\n")
    
    structure_ok = check_project_structure()
    syntax_ok = validate_python_files()
    requirements_ok = check_requirements()
    
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    
    print(f"\n  Project Structure:  {'✓ PASS' if structure_ok else '✗ FAIL'}")
    print(f"  Python Syntax:      {'✓ PASS' if syntax_ok else '✗ FAIL'}")
    print(f"  Requirements:       {'✓ PASS' if requirements_ok else '✗ FAIL'}")
    
    if structure_ok and syntax_ok and requirements_ok:
        print("\n" + "=" * 70)
        print("  ✓ ALL VALIDATIONS PASSED")
        print("=" * 70)
        print("\nNext steps:")
        print("  1. Install dependencies: pip install -r requirements.txt")
        print("  2. Run examples: python examples.py")
        print("  3. Process a video: python run.py --video data/your_video.mp4 --full")
        print()
        return 0
    else:
        print("\n" + "=" * 70)
        print("  ✗ SOME VALIDATIONS FAILED")
        print("=" * 70 + "\n")
        return 1


if __name__ == '__main__':
    exit(main())
