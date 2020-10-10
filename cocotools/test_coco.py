from unittest import TestCase
from . import cocotools

class Test(TestCase):
    def test_promise_set(self):
        self.assertSetEqual(cocotools.promise_set('abc'), {'abc'})
        self.assertSetEqual(cocotools.promise_set(123), {123})
        self.assertSetEqual(cocotools.promise_set({1, 2, 'a'}), {1, 2, 'a'})
        self.assertSetEqual(cocotools.promise_set([1, 2, 'a']), {1, 2, 'a'})
        self.assertSetEqual(cocotools.promise_set(), set())
        self.assertSetEqual(cocotools.promise_set(None), set())
