"""Offline integrity, metric, blinding, and localhost HTTP boundary tests."""
import copy
import hashlib
import http.client
import json
from pathlib import Path
import sys
import tempfile
import threading
import unittest

from scripts.human.human_alignment import (RUBRIC_VERSION, canonical, evaluate, freeze_manifest,
                             preference_accuracy, spearman, validate_manifest)
from scripts.human.review_server import ReviewStore, make_handler
from http.server import ThreadingHTTPServer


def result_rows(root=Path("/video-data"), groups=8):
    return [{"sample_id": f"pair_{group}/video_{video}",
             "video_path": str(root / f"pair_{group}" / f"video_{video}.mp4"),
             "prompt": f"Move object for task {group}. <script>unsafe()</script>",
             "status": "success", "new_score": .12, "old_score": 1.,
             "details": {"secret_model_reason": "Model explanation must not reach UI."}}
            for group in range(groups) for video in range(2)]


def manifest_for(root=Path("/video-data")):
    return freeze_manifest(result_rows(root), {"pair_0/video_0"}, seed="test", heldout_fraction=.5)


def human_label(manifest, sid, score, rater="alice", **extra):
    return {"sample_id": sid, "rater_id": rater, "score": score, "label_source": "human",
            "independent": True, "blinded": True, "manifest_sha256": manifest["manifest_sha256"],
            "rubric_version": RUBRIC_VERSION, **extra}


class FrozenSplitTests(unittest.TestCase):
    def test_deterministic_input_order_pair_grouping_and_reserved_partner(self):
        rows = result_rows()
        a = freeze_manifest(rows, {"pair_0/video_0"}, seed="abc")
        b = freeze_manifest(list(reversed(rows)), {"pair_0/video_0"}, seed="abc")
        self.assertEqual(a, b)
        groups = {}
        for sample in a["samples"]:
            groups.setdefault(sample["pair_id"], set()).add(sample["split"])
        self.assertTrue(all(len(split) == 1 for split in groups.values()))
        self.assertEqual(groups["pair_0"], {"development"})
        self.assertNotIn("new_score", json.dumps(a))
        self.assertNotIn("secret_model_reason", json.dumps(a))

    def test_latest_error_is_not_an_evaluated_case(self):
        rows = result_rows()
        rows.append({"sample_id": "pair_7/video_1", "status": "error"})
        manifest = freeze_manifest(rows, {"pair_0/video_0"})
        self.assertEqual(manifest["sample_count"], 15)
        self.assertNotIn("pair_7/video_1", [r["sample_id"] for r in manifest["samples"]])

    def test_missing_reserved_or_identity_mismatch_fails(self):
        with self.assertRaises(ValueError):
            freeze_manifest(result_rows(), {"not_present/video_0"})
        rows = result_rows()
        rows[0]["video_path"] = "/video-data/different_pair/video_0.mp4"
        with self.assertRaises(ValueError):
            freeze_manifest(rows, set())

    def test_digest_and_group_leakage_are_checked(self):
        manifest = manifest_for()
        manifest["samples"][0]["prompt"] = "modified after freeze"
        with self.assertRaisesRegex(ValueError, "digest"):
            validate_manifest(manifest)
        manifest = manifest_for()
        manifest["samples"][1]["split"] = "heldout"
        manifest["manifest_sha256"] = hashlib.sha256(canonical({k: v for k, v in manifest.items() if k != "manifest_sha256"})).hexdigest()
        with self.assertRaisesRegex(ValueError, "leaks"):
            validate_manifest(manifest)


class MetricsTests(unittest.TestCase):
    def setUp(self):
        self.manifest = manifest_for()
        self.heldout = [r["sample_id"] for r in self.manifest["samples"] if r["split"] == "heldout"]

    def test_no_humans_or_ai_proxies_cannot_establish_alignment(self):
        sid = self.heldout[0]
        labels = [human_label(self.manifest, sid, .9, label_source="ai_proxy")]
        report = evaluate(self.manifest, labels, [{"sample_id": sid, "new_score": .9}])
        self.assertEqual(report["eligible_human_annotations"], 0)
        self.assertFalse(report["human_alignment_achieved"])
        self.assertEqual(report["evidence_status"], "insufficient_independent_heldout_human_labels")
        self.assertEqual(report["excluded_annotations"], {"non_human_source": 1})

    def test_label_provenance_independence_and_numeric_validation(self):
        mutations = [{"independent": False}, {"blinded": False}, {"manifest_sha256": "other"},
                     {"score": float("nan")}, {"score": True}, {"score": 1.1}]
        labels = [human_label(self.manifest, sid, .5, **mutation) if "score" not in mutation
                  else human_label(self.manifest, sid, mutation["score"])
                  for sid, mutation in zip(self.heldout, mutations)]
        report = evaluate(self.manifest, labels, [])
        self.assertEqual(report["eligible_human_annotations"], 0)
        self.assertEqual(sum(report["excluded_annotations"].values()), len(mutations))

    def test_correct_mae_rank_and_interrater_metrics(self):
        labels, predictions = [], []
        for sid, (a, b, prediction) in zip(self.heldout[:2], [(0., .2, .2), (.8, 1., .8)]):
            labels.extend([human_label(self.manifest, sid, a), human_label(self.manifest, sid, b, "bob")])
            predictions.append({"sample_id": sid, "new_score": prediction})
        report = evaluate(self.manifest, labels, predictions)
        heldout = report["by_split"]["heldout"]
        self.assertAlmostEqual(heldout["all_labeled_model_comparison"]["mae"], .1)
        self.assertAlmostEqual(heldout["all_labeled_model_comparison"]["spearman"], 1)
        self.assertAlmostEqual(heldout["inter_rater"]["mae"], .2)
        self.assertEqual(heldout["samples_with_two_or_more_raters"], 2)
        self.assertEqual(heldout["all_labeled_model_comparison"]["within_original_pair_preference"]["strict_accuracy"], 1)
        self.assertFalse(report["human_alignment_achieved"])

    def test_rater_revision_revokes_old_independence(self):
        sid = self.heldout[0]
        labels = [human_label(self.manifest, sid, .8), human_label(self.manifest, sid, .6, independent=False)]
        self.assertEqual(evaluate(self.manifest, labels, [])["eligible_human_annotations"], 0)

    def test_needs_review_is_provisional_and_counts_against_usable_coverage(self):
        sid = self.heldout[0]
        labels = [human_label(self.manifest, sid, .8)]
        predictions = [{"sample_id": sid, "new_score": .8, "status": "needs_review"}]
        section = evaluate(self.manifest, labels, predictions)["by_split"]["heldout"]
        self.assertEqual(section["latest_model_status_counts"]["needs_review"], 1)
        self.assertEqual(section["usable_score_coverage"], 0)
        self.assertIsNone(section["all_labeled_model_comparison"]["mae"])
        self.assertEqual(section["provisional_needs_review_comparison_diagnostic_only"]["mae"], 0)

    def test_repeats_require_explicit_uncached_matching_revision(self):
        sid = self.heldout[0]
        def pred(run, score, **extra):
            return {"sample_id": sid, "new_score": score, "run_id": run, "evaluator_version": "v2", "cache_reused": False, **extra}
        rows = [pred("r1", .6), pred("r2", .8), pred("r3", .1, cache_reused=True), pred("r4", .2, evaluator_version="v1")]
        with self.assertRaisesRegex(ValueError, "Multiple model run"):
            evaluate(self.manifest, [], rows)
        stability = evaluate(self.manifest, [], rows, model_run="r1")["repeatability"]
        self.assertEqual(stability["sample_version_groups_with_at_least_two_uncached_runs"], 1)
        self.assertAlmostEqual(stability["mean_within_sample_std"], .1)
        rows[1].pop("cache_reused")
        self.assertEqual(evaluate(self.manifest, [], rows, model_run="r1")["repeatability"]["sample_version_groups_with_at_least_two_uncached_runs"], 0)

    def test_constant_ranks_and_human_ties_not_false_perfect_agreement(self):
        self.assertIsNone(spearman([1, 1, 1], [0, .5, 1]))
        self.assertAlmostEqual(spearman([0, 0, 1], [1, 1, 0]), -1)
        data = [("a", "p", .1, .5), ("b", "p", .9, .5), ("c", "q", .9, 1.)]
        within = preference_accuracy(data, True)
        self.assertEqual(within["non_tied_human_pairs"], 1)
        self.assertEqual(within["strict_accuracy"], 0)
        self.assertEqual(within["accuracy_half_credit_model_ties"], .5)


class ReviewBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.videos = self.root / "videos"
        self.videos.mkdir()
        self.manifest = manifest_for(self.videos)
        for row in self.manifest["samples"]:
            path = Path(row["video_path"])
            path.parent.mkdir(exist_ok=True)
            path.write_bytes(b"0123456789")
        self.store = ReviewStore(self.manifest, self.root / "labels.jsonl", self.videos, "all")
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(self.store))
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.port = self.server.server_address[1]

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)
        self.tmp.cleanup()

    def request(self, method, path, body=None, headers=None):
        client = http.client.HTTPConnection("127.0.0.1", self.port, timeout=2)
        client.request(method, path, body=body, headers=headers or {})
        response = client.getresponse()
        result = response.status, dict(response.getheaders()), response.read()
        client.close()
        return result

    def payload(self):
        return {"review_id": next(iter(self.store.samples)), "rater_id": "alice", "label_source": "human",
                "independent": True, "score": .75, "task_completion": 1, "physics_severity": "minor",
                "confidence": "high", "notes": "Task completed at 3.1 seconds; small edge flicker.", "watched_full_video": True}

    def test_public_ui_only_contains_allowlisted_blind_fields(self):
        status, _, body = self.request("GET", "/api/items?rater=alice")
        self.assertEqual(status, 200)
        data = json.loads(body)
        self.assertEqual(set(data["items"][0]), {"review_id", "prompt", "video_url"})
        for forbidden in ("old_score", "new_score", "secret_model_reason", "video_path", "pair_0"):
            self.assertNotIn(forbidden, body.decode())
        self.assertEqual(self.request("GET", "/labels.jsonl")[0], 404)
        self.assertEqual(self.request("GET", "/video/../../etc/passwd")[0], 404)

    def test_host_and_csrf_reject_cross_origin_writes(self):
        self.assertEqual(self.request("GET", "/", headers={"Host": "evil.example"})[0], 403)
        body = json.dumps(self.payload())
        headers = {"Content-Type": "application/json"}
        self.assertEqual(self.request("POST", "/api/labels", body, headers)[0], 403)
        headers.update({"X-Review-CSRF": self.store.csrf, "Origin": "https://evil.example"})
        self.assertEqual(self.request("POST", "/api/labels", body, headers)[0], 403)
        self.assertFalse(self.store.labels_path.exists())

    def test_valid_save_export_resume_and_nonhuman_rejection(self):
        payload = self.payload()
        headers = {"Content-Type": "application/json", "X-Review-CSRF": self.store.csrf}
        self.assertEqual(self.request("POST", "/api/labels", json.dumps(payload), headers)[0], 200)
        saved = json.loads(self.store.labels_path.read_text())
        self.assertEqual(saved["manifest_sha256"], self.manifest["manifest_sha256"])
        self.assertEqual(saved["label_source"], "human")
        self.assertTrue(saved["blinded"])
        queue = self.store.queue("alice")
        self.assertEqual(queue["completed"], 1)
        self.assertNotIn(payload["review_id"], [item["review_id"] for item in queue["items"]])
        payload["label_source"] = "ai_proxy"
        self.assertEqual(self.request("POST", "/api/labels", json.dumps(payload), headers)[0], 400)

    def test_video_range_and_symlink_root_escape(self):
        review_id = next(iter(self.store.samples))
        status, headers, body = self.request("GET", "/video/" + review_id, headers={"Range": "bytes=2-5"})
        self.assertEqual((status, body), (206, b"2345"))
        self.assertEqual(headers["Content-Range"], "bytes 2-5/10")
        self.assertEqual(self.request("GET", "/video/" + review_id, headers={"Range": "bytes=100-"})[0], 416)
        video = Path(self.store.samples[review_id]["video_path"])
        video.unlink()
        outside = self.root / "private.mp4"
        outside.write_bytes(b"not served")
        video.symlink_to(outside)
        self.assertEqual(self.request("GET", "/video/" + review_id)[0], 400)

    def test_development_server_does_not_serve_heldout(self):
        restricted = ReviewStore(self.manifest, self.root / "labels.jsonl", self.videos)
        heldout = next(r for r in self.manifest["samples"] if r["split"] == "heldout")
        with self.assertRaises(KeyError):
            restricted.video_path(heldout["review_id"])


if __name__ == "__main__":
    unittest.main()
