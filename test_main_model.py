import unittest

import numpy as np
import pandas as pd

from main_model import ModelLR


class MyTestCase(unittest.TestCase):
    def setUp(self):
        self.model = ModelLR()
        self.model.data_modification()

    def test_data_modification(self):
        self.assertEqual(len(self.model.train), 2318)
        self.assertEqual(len(self.model.test), 122)
        self.assertEqual(len(self.model.target), 2318)
        self.assertEqual(len(self.model.sub_file), 122)
        self.assertFalse("Unnamed: 0" in self.model.df.columns)
        self.assertFalse("period" in self.model.df.columns)

    def test_one_hot_coding(self):
        self.assertEqual(len(self.model.df.columns), 1543)

    def test_count_columns(self):
        self.assertEqual(len(self.model.cat_columns), 0)
        self.assertEqual(len(self.model.num_columns), 1543)

    def test_breakdown_data(self):
        self.assertEqual(len(self.model.x_train_), 1854)
        self.assertEqual(len(self.model.x_val), 464)
        self.assertEqual(len(self.model.y_train_), 1854)
        self.assertEqual(len(self.model.y_val), 464)

    def test_cross_validation(self):
        self.assertTrue(
                        np.array_equal(
                                pd.DataFrame(self.model.df_cv_linreg.mean()[2:]).sort_index(
                                        inplace = True
                                                                   ),
                        pd.DataFrame(
                                ["0.327251", " -0.00710", "-0.046105", "-0.508162"],
                                ["test_R2", "test_-MSE", "test_-MAE", "test_Max"],
            ).sort_index(inplace=True),
                    )
                )

    def test_get_regularization(self):
        self.assertEqual("%.1f" % self.model.ms, "0.1")
        self.assertEqual("%.1f" % self.model.rm, "0.1")
        self.assertEqual("%.4f" % self.model.r2, "0.4171")


if __name__ == "__main__":
    unittest.main()
# coverage run test_main_model.py
# coverage report -m - процент покрытия тестами
