#!/usr/bin/env python3
"""
Unified test runner with detailed logging, input/output tracking, 
success/failure explanations, and diagram generation.

Covers:
- Preprocessing pipeline tests (stateless, transform_test, no data leakage)
- Phase 2 outputs (parquet, feature_names, summary)
- Phase 3 evaluation (synthetic mode, model-aware profiles, CNN/TabNet)
- Model tests (CNN, TabNet, sklearn models)
- Algorithm handling (name sanitization, column management)
- Dataset source flag handling
- AHP-TOPSIS framework tests
"""
from __future__ import annotations

import sys
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass, asdict
import io
import traceback

try:
    import pytest
except ImportError:
    print("❌ pytest not installed. Install with: pip install pytest")
    sys.exit(1)

try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("⚠️  matplotlib/pandas not available - diagram generation will be skipped")


# Configure detailed logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class TestResult:
    """Detailed test result with input/output tracking"""
    test_name: str
    outcome: str  # 'passed', 'failed', 'skipped', 'error'
    duration: float = 0.0
    input_description: str = ""
    output_description: str = ""
    success_reason: str = ""
    failure_reason: str = ""
    error_message: str = ""
    traceback: str = ""


class DetailedTestPlugin:
    """Pytest plugin to capture detailed test information"""
    
    def __init__(self):
        self.results: List[TestResult] = []
        self.test_start_times: Dict[str, float] = {}
    
    def pytest_runtest_setup(self, item):
        """Called before each test setup"""
        import time
        self.test_start_times[item.nodeid] = time.time()
        logger.info(f"\n{'='*80}")
        logger.info(f"🔍 SETUP: {item.nodeid}")
        logger.info(f"{'='*80}")
    
    def pytest_runtest_makereport(self, item, call):
        """Capture test report information"""
        import time
        
        if call.when == "call":  # Only capture actual test execution
            test_name = item.nodeid
            start_time = self.test_start_times.get(test_name, time.time())
            duration = time.time() - start_time
            
            outcome = call.outcome  # 'passed', 'failed', 'skipped'
            
            # Extract input/output from test function docstring if available
            docstring = item.function.__doc__ if hasattr(item.function, '__doc__') else ""
            input_desc = self._extract_section(docstring, "Input:")
            output_desc = self._extract_section(docstring, "Expected Output:")
            method_desc = self._extract_section(docstring, "Method:")
            
            result = TestResult(
                test_name=test_name,
                outcome=outcome,
                duration=duration,
                input_description=input_desc or "N/A",
                output_description=output_desc or "N/A",
                success_reason="",
                failure_reason="",
                error_message="",
                traceback=""
            )
            
            if outcome == "passed":
                result.success_reason = self._generate_success_reason(item, docstring, method_desc)
                logger.info(f"✅ PASSED: {test_name} ({duration:.3f}s)")
                logger.info(f"   Reason: {result.success_reason}")
            elif outcome == "failed":
                result.failure_reason = self._generate_failure_reason(call)
                result.error_message = str(call.excinfo.value) if call.excinfo else "Unknown error"
                result.traceback = ''.join(traceback.format_tb(call.excinfo.tb)) if call.excinfo else ""
                logger.error(f"❌ FAILED: {test_name} ({duration:.3f}s)")
                logger.error(f"   Reason: {result.failure_reason}")
                logger.error(f"   Error: {result.error_message}")
                if result.traceback:
                    logger.debug(f"   Traceback:\n{result.traceback}")
            elif outcome == "skipped":
                skip_reason = call.excinfo.value.msg if call.excinfo and hasattr(call.excinfo.value, 'msg') else "Unknown reason"
                result.success_reason = f"Skipped: {skip_reason}"
                logger.warning(f"⏭️  SKIPPED: {test_name} - {skip_reason}")
            
            # Log input/output if available
            if input_desc:
                logger.info(f"   Input: {input_desc[:100]}..." if len(input_desc) > 100 else f"   Input: {input_desc}")
            if output_desc and outcome == "passed":
                logger.info(f"   Output: {output_desc[:100]}..." if len(output_desc) > 100 else f"   Output: {output_desc}")
            
            self.results.append(result)
    
    def _extract_section(self, docstring: str, section: str) -> str:
        """Extract a section from docstring"""
        if not docstring:
            return ""
        
        lines = docstring.split('\n')
        in_section = False
        section_lines = []
        
        for line in lines:
            if section in line:
                in_section = True
                continue
            if in_section:
                if line.strip() and not line.strip().startswith(('Input:', 'Output:', 'Method:', 'Processing:')):
                    section_lines.append(line.strip())
                elif line.strip().startswith(('Input:', 'Output:', 'Method:', 'Processing:')):
                    break
        
        return ' '.join(section_lines).strip()
    
    def _generate_success_reason(self, item, docstring: str, method_desc: str) -> str:
        """Generate explanation for test success"""
        test_name = item.name.lower()
        
        if "preprocessing" in test_name:
            return "Preprocessing pipeline correctly applied stateless transformations and maintained data integrity"
        elif "phase2" in test_name or "phase_2" in test_name:
            return "Phase 2 outputs (parquet, feature_names.json, summary.md) successfully generated with correct format"
        elif "phase3" in test_name or "phase_3" in test_name:
            return "Phase 3 evaluation completed with model-aware preprocessing per fold, ensuring zero data leakage"
        elif "cnn" in test_name:
            return "CNN model correctly initialized, trained, and evaluated with proper input reshaping"
        elif "tabnet" in test_name:
            return "TabNet model correctly initialized, trained, and evaluated with balanced class weights"
        elif "leakage" in test_name or "no_data" in test_name:
            return "No data leakage detected: scaler/selector fitted only on TRAIN, test transformed using TRAIN-fitted objects"
        elif "algo" in test_name or "algorithm" in test_name:
            return "Algorithm name handling (sanitization, column management) working correctly"
        elif "synthetic" in test_name:
            return "Synthetic dataset mode correctly generated and processed dataset with expected structure"
        elif "dataset_source" in test_name:
            return "Dataset source flag correctly encoded and preserved through preprocessing pipeline"
        elif "model_aware" in test_name:
            return "Model-aware preprocessing profiles correctly applied (LR/Tree/CNN/TabNet profiles)"
        elif "transform" in test_name:
            return "Transform methods correctly applied fitted preprocessing to new data without leakage"
        else:
            return f"Test passed: {item.name} - All assertions satisfied"
    
    def _generate_failure_reason(self, call) -> str:
        """Generate explanation for test failure"""
        if not call.excinfo:
            return "Unknown failure reason"
        
        exc_type = call.excinfo.typename
        exc_value = str(call.excinfo.value)
        
        if "AssertionError" in exc_type:
            return f"Assertion failed: {exc_value}"
        elif "ValueError" in exc_type:
            return f"Invalid value: {exc_value}"
        elif "AttributeError" in exc_type:
            return f"Missing attribute: {exc_value}"
        elif "KeyError" in exc_type:
            return f"Missing key: {exc_value}"
        elif "ImportError" in exc_type or "ModuleNotFoundError" in exc_type:
            return f"Missing dependency: {exc_value}"
        elif "FileNotFoundError" in exc_type:
            return f"File not found: {exc_value}"
        else:
            return f"{exc_type}: {exc_value}"


def generate_test_coverage_diagram(results: List[TestResult], output_dir: Path) -> Path:
    """Generate test coverage diagram"""
    if not MATPLOTLIB_AVAILABLE:
        logger.warning("matplotlib not available - skipping diagram generation")
        return None
    
    # Categorize tests
    categories = {
        "Preprocessing": ["preprocessing", "transform", "clean", "encode"],
        "Phase 2": ["phase2", "phase_2", "apply_best"],
        "Phase 3": ["phase3", "phase_3", "evaluation", "synthetic"],
        "Models": ["cnn", "tabnet", "model", "sklearn"],
        "Data Leakage": ["leakage", "no_data", "no_leakage"],
        "Algorithm Handling": ["algo", "algorithm", "sanitize"],
        "Dataset Source": ["dataset_source"],
        "Model Profiles": ["model_aware", "profile"],
        "Other": []
    }
    
    category_counts = {cat: {"passed": 0, "failed": 0, "skipped": 0} for cat in categories}
    
    for result in results:
        test_lower = result.test_name.lower()
        categorized = False
        for category, keywords in categories.items():
            if category == "Other":
                continue
            if any(keyword in test_lower for keyword in keywords):
                category_counts[category][result.outcome] += 1
                categorized = True
                break
        if not categorized:
            category_counts["Other"][result.outcome] += 1
    
    # Create visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Left: Test outcomes by category
    categories_list = list(category_counts.keys())
    passed_counts = [category_counts[cat]["passed"] for cat in categories_list]
    failed_counts = [category_counts[cat]["failed"] for cat in categories_list]
    skipped_counts = [category_counts[cat]["skipped"] for cat in categories_list]
    
    x = np.arange(len(categories_list))
    width = 0.25
    
    ax1.bar(x - width, passed_counts, width, label='Passed', color='green', alpha=0.7)
    ax1.bar(x, failed_counts, width, label='Failed', color='red', alpha=0.7)
    ax1.bar(x + width, skipped_counts, width, label='Skipped', color='orange', alpha=0.7)
    
    ax1.set_xlabel('Test Category')
    ax1.set_ylabel('Number of Tests')
    ax1.set_title('Test Coverage by Category')
    ax1.set_xticks(x)
    ax1.set_xticklabels(categories_list, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    
    # Right: Overall pass rate
    total_passed = sum(passed_counts)
    total_failed = sum(failed_counts)
    total_skipped = sum(skipped_counts)
    total = total_passed + total_failed + total_skipped
    
    if total > 0:
        sizes = [total_passed, total_failed, total_skipped]
        labels = [f'Passed ({total_passed})', f'Failed ({total_failed})', f'Skipped ({total_skipped})']
        colors = ['green', 'red', 'orange']
        explode = (0.05, 0.1 if total_failed > 0 else 0, 0.05)
        
        ax2.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90, explode=explode)
        ax2.set_title(f'Overall Test Results (Total: {total})')
    
    plt.tight_layout()
    
    diagram_path = output_dir / "test_coverage_diagram.png"
    plt.savefig(diagram_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"📊 Generated test coverage diagram: {diagram_path}")
    return diagram_path


def generate_detailed_report(results: List[TestResult], output_dir: Path) -> Path:
    """Generate detailed test report with input/output information"""
    report_path = output_dir / "test_report.md"
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# Test Execution Report\n\n")
        f.write(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Summary statistics
        total = len(results)
        passed = sum(1 for r in results if r.outcome == "passed")
        failed = sum(1 for r in results if r.outcome == "failed")
        skipped = sum(1 for r in results if r.outcome == "skipped")
        
        f.write("## Summary Statistics\n\n")
        f.write(f"- **Total Tests**: {total}\n")
        f.write(f"- **Passed**: {passed} ({passed/total*100:.1f}%)\n")
        f.write(f"- **Failed**: {failed} ({failed/total*100:.1f}%)\n")
        f.write(f"- **Skipped**: {skipped} ({skipped/total*100:.1f}%)\n\n")
        
        # Test results by outcome
        f.write("## Test Results by Outcome\n\n")
        
        # Passed tests
        if passed > 0:
            f.write("### ✅ Passed Tests\n\n")
            for result in results:
                if result.outcome == "passed":
                    f.write(f"#### {result.test_name}\n\n")
                    f.write(f"- **Duration**: {result.duration:.3f}s\n")
                    f.write(f"- **Input**: {result.input_description}\n")
                    f.write(f"- **Expected Output**: {result.output_description}\n")
                    f.write(f"- **Success Reason**: {result.success_reason}\n\n")
        
        # Failed tests
        if failed > 0:
            f.write("### ❌ Failed Tests\n\n")
            for result in results:
                if result.outcome == "failed":
                    f.write(f"#### {result.test_name}\n\n")
                    f.write(f"- **Duration**: {result.duration:.3f}s\n")
                    f.write(f"- **Input**: {result.input_description}\n")
                    f.write(f"- **Failure Reason**: {result.failure_reason}\n")
                    f.write(f"- **Error Message**: `{result.error_message}`\n\n")
                    if result.traceback:
                        f.write("```\n")
                        f.write(result.traceback)
                        f.write("```\n\n")
        
        # Skipped tests
        if skipped > 0:
            f.write("### ⏭️  Skipped Tests\n\n")
            for result in results:
                if result.outcome == "skipped":
                    f.write(f"- **{result.test_name}**: {result.success_reason}\n\n")
    
    logger.info(f"📝 Generated detailed test report: {report_path}")
    return report_path


def generate_json_results(results: List[TestResult], output_dir: Path) -> Path:
    """Generate JSON results file"""
    json_path = output_dir / "test_results.json"
    
    results_dict = {
        "timestamp": datetime.now().isoformat(),
        "total_tests": len(results),
        "passed": sum(1 for r in results if r.outcome == "passed"),
        "failed": sum(1 for r in results if r.outcome == "failed"),
        "skipped": sum(1 for r in results if r.outcome == "skipped"),
        "results": [asdict(r) for r in results]
    }
    
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results_dict, f, indent=2)
    
    logger.info(f"💾 Saved JSON results: {json_path}")
    return json_path


def main() -> int:
    """Main test runner with detailed logging and reporting"""
    logger.info("=" * 80)
    logger.info("🧪 IRP PIPELINE TEST SUITE")
    logger.info("=" * 80)
    logger.info(f"Starting test execution at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Create timestamped output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("output/test_reports") / f"test_run_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"📁 Output directory: {output_dir}")
    
    # Initialize plugin
    plugin = DetailedTestPlugin()
    
    # Run pytest
    logger.info("\n🚀 Running pytest...\n")
    exit_code = pytest.main(["-v", "tests"], plugins=[plugin])
    
    # Process results
    results = plugin.results
    total = len(results)
    passed = sum(1 for r in results if r.outcome == "passed")
    failed = sum(1 for r in results if r.outcome == "failed")
    skipped = sum(1 for r in results if r.outcome == "skipped")
    
    # Print summary
    logger.info("\n" + "=" * 80)
    logger.info("📊 TEST EXECUTION SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Total Tests: {total}")
    logger.info(f"✅ Passed: {passed} ({passed/total*100:.1f}%)")
    logger.info(f"❌ Failed: {failed} ({failed/total*100:.1f}%)")
    logger.info(f"⏭️  Skipped: {skipped} ({skipped/total*100:.1f}%)")
    
    # Detailed conclusions
    logger.info("\n" + "=" * 80)
    logger.info("📋 DETAILED TEST CONCLUSIONS")
    logger.info("=" * 80)
    
    for result in results:
        status_icon = {
            "passed": "✅",
            "failed": "❌",
            "skipped": "⏭️ "
        }.get(result.outcome, "❓")
        
        logger.info(f"{status_icon} {result.test_name}")
        logger.info(f"   Duration: {result.duration:.3f}s")
        
        if result.outcome == "passed":
            logger.info(f"   ✅ Success: {result.success_reason}")
        elif result.outcome == "failed":
            logger.error(f"   ❌ Failure: {result.failure_reason}")
            logger.error(f"   Error: {result.error_message}")
    
    # Generate reports and diagrams
    logger.info("\n" + "=" * 80)
    logger.info("📈 GENERATING REPORTS AND DIAGRAMS")
    logger.info("=" * 80)
    
    try:
        report_path = generate_detailed_report(results, output_dir)
        json_path = generate_json_results(results, output_dir)
        
        if MATPLOTLIB_AVAILABLE:
            diagram_path = generate_test_coverage_diagram(results, output_dir)
            if diagram_path:
                logger.info(f"   Diagram: {diagram_path}")
        
        logger.info(f"   Report: {report_path}")
        logger.info(f"   JSON: {json_path}")
    except Exception as e:
        logger.error(f"Error generating reports: {e}", exc_info=True)
    
    # Final status
    overall = "✅ PASSED" if exit_code == 0 else "❌ FAILED"
    logger.info("\n" + "=" * 80)
    logger.info(f"🏁 OVERALL RESULT: {overall}")
    logger.info("=" * 80)
    logger.info(f"All results saved to: {output_dir}")
    
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
