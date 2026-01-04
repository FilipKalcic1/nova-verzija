import pytest
import sys
import logging

# Configure logging
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
log = logging.getLogger(__name__)

def main():
    """
    Programmatically runs Pytest with strict settings.
    This is a workaround for environments where shell commands are limited.
    """
    log.info("--- Starting Pytest with strict settings ---")
    log.info("Warnings will be treated as errors.")
    log.info("A summary of skipped tests will be displayed.")

    # -W error: treat warnings as errors
    # -rS: show a summary of skipped tests
    # --showlocals: show local variables in tracebacks (useful for debugging)
    # tests/: specify the directory to run tests from
    args = ["-W", "error", "-rS", "--showlocals", "tests/"]
    
    try:
        exit_code = pytest.main(args)
    except Exception as e:
        log.error(f"An exception occurred while running pytest: {e}", exc_info=True)
        sys.exit(1)

    log.info(f"--- Pytest finished with exit code: {exit_code} ---")

    if exit_code == 0:
        log.info("✅ All tests passed successfully without warnings.")
    else:
        log.warning("‼️ Some tests failed, had warnings, or were skipped.")
        log.warning("Please review the full output above for details.")
    
    # Exit with the same code to signal success/failure
    sys.exit(exit_code)

if __name__ == "__main__":
    main()
