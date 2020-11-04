from unittest import TestCase
from .distribution_analyze import parse_channel_from_img_name

class Test(TestCase):
    def test_parse_channel_from_img_name(self):
        test_examples = (
            ('WANDA_bj_tzwd_face_20200127_ch15013_20200127191219.mp4.cut.mp4_225.jpg', ('WANDA_bj_tzwd', 'ch15013')),
            ('WANDA_bj_tzwd_face_20200127_ch09008_20200127092956.mp4.cut.mp4_25.jpg', ('WANDA_bj_tzwd', 'ch09008')),
            ('CC_tianjin_888_face_20191112_ch01003_20191112171104.mp4.cut.mp4_25.jpg', ('CC_tianjin_888', 'ch01003')),
            ('AEGEAN_suzhou_wujiang_face_20190928_ch02020_20190928202117.mp4.cut.mp4_1825.jpg',
             ('AEGEAN_suzhou_wujiang', 'ch02020')),
            ("AFU_ch01001_20190204153452_13000.jpg", ('AFU', 'ch01001'))
        )
        for input, answer in test_examples:
            self.assertEqual(parse_channel_from_img_name(input), answer, f'parse_img_name({input})->{parse_channel_from_img_name(input)} != {answer}')
