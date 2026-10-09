#!/usr/bin/env python3
"""Parse and echo the assessment's 16-byte UART protocol."""
import argparse
import os
import select
import struct
import termios
import time
import tty

PACKET = struct.Struct('>Bh')
SIZE = 16


def encode(value, big_endian, little_endian):
    return b'\xff' + PACKET.pack(value, big_endian) + struct.pack('<h', little_endian) + bytes(9) + b'\r'


def decode(packet):
    if len(packet) != SIZE or packet[0] != 255 or packet[15] != 13 or any(packet[6:15]):
        raise ValueError('invalid UART packet')
    value, big_endian = PACKET.unpack(packet[1:4])
    return value, big_endian, struct.unpack('<h', packet[4:6])[0]


class Parser:
    def __init__(self):
        self.buffer = bytearray()

    def feed(self, data):
        self.buffer.extend(data)
        packets = []
        while self.buffer:
            start = self.buffer.find(b'\xff')
            if start < 0:
                self.buffer.clear()
                break
            del self.buffer[:start]
            if len(self.buffer) < SIZE:
                break
            try:
                values = decode(self.buffer[:SIZE])
            except ValueError:
                del self.buffer[0]
                continue
            packets.append(values)
            del self.buffer[:SIZE]
        return packets


def write_all(fd, data):
    view = memoryview(data)
    while view:
        try:
            written = os.write(fd, view)
        except BlockingIOError:
            select.select([], [fd], [], 1)
            continue
        if written == 0:
            raise OSError('serial write returned zero bytes')
        view = view[written:]


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--port', default='/tmp/test_com')
    cli.add_argument('--count', type=int, default=0, help='stop after N packets; 0 means unlimited')
    cli.add_argument('--timeout', type=float, default=10, help='receive timeout in seconds')
    args = cli.parse_args()
    if args.count < 0 or args.timeout <= 0:
        cli.error('count must be nonnegative and timeout must be positive')
    fd = os.open(args.port, os.O_RDWR | os.O_NOCTTY | os.O_NONBLOCK)
    original = termios.tcgetattr(fd)
    received = 0
    try:
        tty.setraw(fd, termios.TCSANOW)
        attrs = termios.tcgetattr(fd)
        attrs[4] = attrs[5] = termios.B115200
        attrs[2] |= termios.CLOCAL | termios.CREAD
        termios.tcsetattr(fd, termios.TCSANOW, attrs)
        parser = Parser()
        deadline = time.monotonic() + args.timeout
        print(f'Opened {args.port}: 115200 baud, 8N1', flush=True)
        while not args.count or received < args.count:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError('no valid packet received before timeout')
            ready, _, _ = select.select([fd], [], [], remaining)
            if not ready:
                continue
            try:
                data = os.read(fd, 4096)
            except BlockingIOError:
                continue
            if not data:
                raise EOFError('serial peer disconnected')
            for values in parser.feed(data):
                write_all(fd, encode(*values))
                received += 1
                print(f'[{received}] uint8={values[0]} int16_be={values[1]} int16_le={values[2]} echoed', flush=True)
                deadline = time.monotonic() + args.timeout
                if args.count and received >= args.count:
                    break
    finally:
        try:
            termios.tcsetattr(fd, termios.TCSANOW, original)
        except termios.error:
            pass
        os.close(fd)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass
    except (OSError, EOFError, TimeoutError) as error:
        raise SystemExit(str(error))
