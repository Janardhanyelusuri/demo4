import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def setup_logging():
    logging.getLogger("uvicorn.access").handlers.clear()
    logging.getLogger("uvicorn.error").handlers.clear()
    uvicorn_access_handler = logging.StreamHandler()
    uvicorn_access_handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
    uvicorn_error_handler = logging.StreamHandler()
    uvicorn_error_handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))

    logging.getLogger("uvicorn.access").addHandler(uvicorn_access_handler)
    logging.getLogger("uvicorn.error").addHandler(uvicorn_error_handler)






