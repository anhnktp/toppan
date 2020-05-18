import unittest
from helpers.cam_data import get_evidence_cams, get_engine_cams


class TestCam(unittest.TestCase):

    def test_camdata(self):
        engine_cams = get_engine_cams()
        # print(engine_cams)
        self.assertIsNotNone(engine_cams)
        self.assertEqual(len(engine_cams), 4)


    def test_evidencedata(self):
        evidence_cams = get_evidence_cams()
        self.assertIsNotNone(evidence_cams)
        self.assertEqual(len(evidence_cams), 4)

if __name__ == '__main__':
    unittest.main()