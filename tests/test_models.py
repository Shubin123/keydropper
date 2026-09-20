import unittest

import _pathfix  # noqa: F401

from keysoundmap import models as M


class TestClassifiers(unittest.TestCase):
    def _toy(self):
        # Two well-separated clusters in 4-D.
        X = [[0, 0, 0, 0], [0.1, 0, 0, 0], [0, 0.1, 0, 0],
             [5, 5, 5, 5], [5.1, 5, 5, 5], [5, 5.1, 5, 5]]
        y = ["a", "a", "a", "b", "b", "b"]
        return X, y

    def test_prototype_classifier_separates_clusters(self):
        X, y = self._toy()
        clf = M.PrototypeClassifier().fit(X, y)
        self.assertEqual(clf.predict([[0.05, 0.05, 0, 0]]), ["a"])
        self.assertEqual(clf.predict([[5.05, 5, 5, 5]]), ["b"])

    def test_prototype_proba_sums_to_one(self):
        X, y = self._toy()
        clf = M.PrototypeClassifier().fit(X, y)
        proba = clf.predict_proba([[0, 0, 0, 0]])[0]
        self.assertAlmostEqual(sum(proba.values()), 1.0, places=6)

    def test_knn_separates_clusters(self):
        X, y = self._toy()
        clf = M.KNNClassifier(k=1).fit(X, y)
        self.assertEqual(clf.predict([[0.05, 0.05, 0, 0]]), ["a"])
        self.assertEqual(clf.predict([[4.9, 5, 5, 5]]), ["b"])

    def test_knn_self_prediction_is_perfect(self):
        X, y = self._toy()
        clf = M.KNNClassifier(k=1).fit(X, y)
        self.assertEqual(M.accuracy(clf.predict(X), y), 1.0)

    def test_infer_n_frames_positive(self):
        from keysoundmap.config import AudioConfig, FeatureConfig
        n = M.infer_n_frames(120.0, FeatureConfig(), AudioConfig())
        self.assertGreater(n, 1)

    def test_accuracy_helper(self):
        self.assertEqual(M.accuracy(["a", "b"], ["a", "a"]), 0.5)
        self.assertEqual(M.accuracy([], []), 0.0)


class TestDeepModelsGuarded(unittest.TestCase):
    def test_torch_models_build_or_raise_cleanly(self):
        from keysoundmap.config import ModelConfig
        try:
            import torch  # noqa: F401
        except Exception:
            with self.assertRaises(ImportError):
                M.build_tiny_key_cnn(10, 40, ModelConfig())
            return
        net = M.build_tiny_key_cnn(11, 40, ModelConfig())
        p = M.count_params(net)
        self.assertGreater(p, 0)
        self.assertLess(p, 500_000, "TinyKeyCNN should stay small/efficient")


if __name__ == "__main__":
    unittest.main()
