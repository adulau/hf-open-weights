import unittest
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

# Keep these unit tests runnable without the optional network dependency. The
# tested iterator only needs duck-typed API/model objects.
hub_module = MagicMock()
hub_module.HfApi = MagicMock
hub_module.ModelCard = MagicMock
hub_errors_module = MagicMock()
hub_errors_module.HfHubHTTPError = type("HfHubHTTPError", (Exception,), {})
hub_errors_module.RepositoryNotFoundError = type(
    "RepositoryNotFoundError", (Exception,), {}
)
sys.modules.setdefault("huggingface_hub", hub_module)
sys.modules.setdefault("huggingface_hub.errors", hub_errors_module)

import hf_open_weights


class RecordingApi:
    def __init__(self, models):
        self.models = models
        self.list_models_kwargs = None

    def list_models(self, **kwargs):
        self.list_models_kwargs = kwargs
        return iter(self.models)


class IterCandidatesTests(unittest.TestCase):
    def test_does_not_request_eager_card_data(self):
        model = SimpleNamespace(
            id="example/model",
            private=False,
            tags=["license:apache-2.0"],
            siblings=[SimpleNamespace(rfilename="model.safetensors")],
        )
        api = RecordingApi([model])

        candidates = list(
            hf_open_weights.iter_candidates(
                api,
                policy="open-weight",
                limit=None,
                since=None,
                sort="last-modified",
            )
        )

        self.assertEqual([(model, {}, ["model.safetensors"])], candidates)
        self.assertEqual(
            {"sort": "lastModified", "limit": None, "full": True},
            api.list_models_kwargs,
        )

    def test_can_start_with_most_starred_models(self):
        api = RecordingApi([])

        list(
            hf_open_weights.iter_candidates(
                api,
                policy="open-weight",
                limit=25,
                since=None,
                sort="most-starred",
            )
        )

        self.assertEqual(
            {"sort": "likes", "limit": 25, "full": True},
            api.list_models_kwargs,
        )

    def test_parser_defaults_to_last_modified_sort(self):
        parser = hf_open_weights.build_parser()

        self.assertEqual("last-modified", parser.parse_args([]).sort)
        self.assertEqual(
            "most-starred",
            parser.parse_args(["--sort", "most-starred"]).sort,
        )


if __name__ == "__main__":
    unittest.main()
