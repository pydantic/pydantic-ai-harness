from __future__ import annotations

import importlib.util

collect_ignore = [] if importlib.util.find_spec('sprites') else ['test_sprites.py']
