import unittest

from test import route_from_filename


class FilenameRouterTest(unittest.TestCase):
    def test_day_image_uses_dora1(self):
        self.assertEqual(route_from_filename('IMG_62'), ('day', 'dora1'))

    def test_night_image_uses_dora(self):
        self.assertEqual(
            route_from_filename('night_IMG_62'), ('night', 'dora'))

    def test_router_accepts_paths_and_case_insensitive_prefix(self):
        self.assertEqual(
            route_from_filename('/dataset/NIGHT_IMG_1.jpg'),
            ('night', 'dora'))


if __name__ == '__main__':
    unittest.main()
