'''
Created on Apr 2, 2020

@author: W. M. Heigl
'''
import unittest
from unsupervised.self_organizing_map import *


class TestSOM(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_init(self):
        shape = (10, 15, 7)
        learning_rate = 0.01
        iterations = 200
        som = SelfOrganizingMap(shape, learning_rate, iterations)
        self.assertIsNotNone(som, 'Object must not be None')
        self.assertEqual(som.shape, shape, "Actual shape not equal to expected")
        self.assertEqual(som.learning_rate, learning_rate, "Actual learning rate not equal to expected")
        self.assertEqual(som.iterations, iterations, "Actual no. of iterations not equal to expected")

    def test_learn(self):
        shape = (1, 0, 0)
        learning_rate = 0.01
        iterations = 200
        som = SelfOrganizingMap(shape, learning_rate, iterations)
        som.learn(data=None)

if __name__ == "__main__":
    # import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
