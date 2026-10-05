"""Every plan.json written by the pipeline must validate against the published schema."""
import glob, json, os
import pytest

jsonschema = pytest.importorskip("jsonschema")
SCHEMA = json.load(open(os.path.join(os.path.dirname(__file__), "..", "roomplan", "schema.json")))
PLANS = glob.glob(os.environ.get("ROOMPLAN_OUT", "out") + "/**/plan.json", recursive=True)


@pytest.mark.skipif(not PLANS, reason="no outputs yet: run python -m roomplan on a capture first")
@pytest.mark.parametrize("path", PLANS)
def test_plan_matches_schema(path):
    jsonschema.validate(json.load(open(path)), SCHEMA)
