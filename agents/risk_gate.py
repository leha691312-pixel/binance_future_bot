"""
Risk Gate - enable_low_remaining_rr=false disables LOW_REMAINING_RR
"""
import zlib, base64
from agents._rg_data1 import B64_PART1
from agents._rg_data2 import B64_PART2
_SRC = zlib.decompress(base64.b64decode(B64_PART1 + B64_PART2)).decode("utf-8")
exec(_SRC, globals())
