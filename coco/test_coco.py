from unittest import TestCase

import coco.utils
from . import coco

class Test(TestCase):
    def test_promise_set(self):
        self.assertSetEqual(coco.utils.promise_set('abc'), {'abc'})
        self.assertSetEqual(coco.utils.promise_set(123), {123})
        self.assertSetEqual(coco.utils.promise_set({1, 2, 'a'}), {1, 2, 'a'})
        self.assertSetEqual(coco.utils.promise_set([1, 2, 'a']), {1, 2, 'a'})
        self.assertSetEqual(coco.utils.promise_set(), set())
        self.assertSetEqual(coco.utils.promise_set(None), set())
