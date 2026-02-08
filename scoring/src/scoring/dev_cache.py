"""Development cache utilities for skipping expensive pipeline stages.

Provides simple pickle-based caching to speed up development iteration.
Enable by passing --cache-dir <path> on the command line.

Cache files are NOT automatically invalidated. Delete the cache directory
to force a fresh run when input data or upstream code changes.
"""

import logging
import os
import pickle
import time


logger = logging.getLogger("birdwatch.dev_cache")
logger.setLevel(logging.INFO)

# Global cache directory. None means caching is disabled.
_cache_dir = None


def configure(cache_dir):
  """Enable caching by setting the cache directory. Creates it if needed."""
  global _cache_dir
  if cache_dir:
    _cache_dir = os.path.abspath(cache_dir)
    os.makedirs(_cache_dir, exist_ok=True)
    logger.info(f"Dev cache enabled: {_cache_dir}")
  else:
    _cache_dir = None


def is_enabled():
  """Return True if caching is currently enabled."""
  return _cache_dir is not None


def get_path(name):
  """Return the full path for a named cache file."""
  if _cache_dir is None:
    return None
  return os.path.join(_cache_dir, f"{name}.pkl")


def save(name, data):
  """Pickle data to a named cache file. No-op if caching is disabled."""
  if not is_enabled():
    return
  path = get_path(name)
  start = time.time()
  with open(path, "wb") as f:
    pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
  elapsed = time.time() - start
  size_mb = os.path.getsize(path) / (1024 * 1024)
  logger.info(f"Saved cache '{name}' ({size_mb:.1f} MB, {elapsed:.1f}s) -> {path}")


def load(name):
  """Load data from a named cache file. Returns None on miss or if disabled."""
  if not is_enabled():
    return None
  path = get_path(name)
  if not os.path.exists(path):
    logger.info(f"Cache miss: '{name}' (no file at {path})")
    return None
  start = time.time()
  with open(path, "rb") as f:
    data = pickle.load(f)
  elapsed = time.time() - start
  logger.info(f"Cache hit: '{name}' (loaded in {elapsed:.1f}s) <- {path}")
  return data


def clear(name=None):
  """Delete a specific cache file, or all cache files if name is None."""
  if not is_enabled():
    return
  if name is not None:
    path = get_path(name)
    if os.path.exists(path):
      os.remove(path)
      logger.info(f"Cleared cache '{name}': {path}")
  else:
    for f in os.listdir(_cache_dir):
      if f.endswith(".pkl"):
        os.remove(os.path.join(_cache_dir, f))
        logger.info(f"Cleared cache file: {f}")
