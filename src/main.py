"""Application entry point.

The entry point deliberately contains only composition and process startup.  HTTP,
application, and infrastructure concerns live in their respective modules.
"""

import uvicorn

from infrastructure.settings.config import get_settings
from presentation.app import create_app

app = create_app()


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=get_settings().app_port)
