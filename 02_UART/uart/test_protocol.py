import unittest
from uart_echo import Parser, decode, encode


class ProtocolTests(unittest.TestCase):
    def test_signed_boundaries(self):
        for values in [(0, 0, 0), (255, -32768, 32767), (127, 32767, -32768), (255, -1, -256)]:
            self.assertEqual(decode(encode(*values)), values)

    def test_every_split_position(self):
        packet = encode(255, -1, -256)
        for split in range(1, 16):
            parser = Parser()
            self.assertEqual(parser.feed(packet[:split]), [])
            self.assertEqual(parser.feed(packet[split:]), [(255, -1, -256)])

    def test_noise_bad_padding_and_concatenated_packets(self):
        bad = bytearray(encode(10, 256, 1))
        bad[6] = 1
        values = [(255, -1, -256), (127, 32767, -32768)]
        parser = Parser()
        self.assertEqual(parser.feed(b'noise' + bad + b'garbage' + b''.join(encode(*v) for v in values)), values)
        self.assertEqual(parser.buffer, b'')

    def test_bad_tail_resynchronization(self):
        bad = bytearray(encode(10, 256, 1))
        bad[-1] = 0
        self.assertEqual(Parser().feed(bad + encode(255, -1, -256)), [(255, -1, -256)])


if __name__ == '__main__':
    unittest.main()
