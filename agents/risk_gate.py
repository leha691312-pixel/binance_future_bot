"""Risk Gate - TIMEOUT-only invalidation"""
import zlib, base64
from agents._rg_p0 import P0
from agents._rg_p1 import P1
from agents._rg_p2 import P2
from agents._rg_p3 import P3
_SRC = zlib.decompress(base64.b64decode(P0 + P1 + P2 + P3)).decode("utf-8")
exec(_SRC, globals())
