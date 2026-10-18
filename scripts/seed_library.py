"""Upload kb/*.md to the Evolus knowledge library and import the OSHA 1910.147 page; then list documents."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from torque.config import get_settings  # noqa: E402
from torque.evolus import library  # noqa: E402
from torque.evolus.client import EvolusClient  # noqa: E402

OSHA = "https://www.osha.gov/laws-regs/regulations/standardnumber/1910/1910.147"

if __name__ == "__main__":
    s = get_settings()
    c = EvolusClient(s)
    if c.mock or not s.evolus_library_id:
        print("EVOLUS_API_KEY / EVOLUS_LIBRARY_ID not set: running in mock mode (nothing uploaded)")
    for p in sorted((ROOT / "kb").glob("*.md")):
        if p.name == "README.md":
            continue
        print("upload", p.name, "->", library.upload_document(c, p.name, p.read_bytes()))
    print("import-url", OSHA, "->", library.import_url(c, OSHA))
    print("documents:", library.list_documents(c))
    print("Poll this script's last line (or the workspace UI) until every document shows as indexed.")
