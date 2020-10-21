from unittest import TestCase
from . import coco

class Test(TestCase):
    def test_promise_set(self):
        self.assertSetEqual(coco.promise_set('abc'), {'abc'})
        self.assertSetEqual(coco.promise_set(123), {123})
        self.assertSetEqual(coco.promise_set({1, 2, 'a'}), {1, 2, 'a'})
        self.assertSetEqual(coco.promise_set([1, 2, 'a']), {1, 2, 'a'})
        self.assertSetEqual(coco.promise_set(), set())
        self.assertSetEqual(coco.promise_set(None), set())
