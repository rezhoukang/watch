"""Local vision platform entry point. Start with: python app.py."""
import logging

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from web import common, status, media, history, detection, datasets, jobs, head_cuts, face_recognition, aggregate

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
app = FastAPI(title="本地视觉训练平台")


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    common.logger.exception("未处理异常：%s %s", request.method, request.url.path)
    return JSONResponse(
        status_code=500,
        content={"detail": f"服务器内部错误（{type(exc).__name__}）：{exc}"},
    )


for routes in (status, media, history, detection, datasets, jobs, head_cuts, face_recognition, aggregate):
    app.include_router(routes.router)

# Register the catch-all static mount after every API router.
common.STATIC_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/", StaticFiles(directory=common.STATIC_DIR, html=True), name="static")

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
