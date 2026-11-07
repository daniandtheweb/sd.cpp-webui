"""sd.cpp-webui - extensions - the API version

An extension declares the API version it targets (API_VERSION = N in its
entry file, mandatory). The app has one version today (LATEST). An
extension that targets a newer one is skipped, and one that declares a
version that doesn't exist fails to load.

When a change breaks what extensions already use, that is a new version:
keep the old behaviour for the extensions that declare the old number
(api.version tells the code which one it is serving).
"""

LATEST = 1

# The generation hooks
HOOKS = ('pre_generation', 'post_generation')

# The generation tabs an extension can add its own UI to
TABS = ('txt2img', 'img2img', 'imgedit', 'any2video')
