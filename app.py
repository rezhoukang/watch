"""Local vision platform entry point. Start with: python app.py."""
import logging

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from web import common, status, media, history, detection, datasets, jobs, head_cuts, face_recognition, aggregate, actions

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
app = FastAPI(title="本地视觉训练平台")


class NoCacheStaticFiles(StaticFiles):
    """静态资源一律不缓存。

    本地工具改前端很频繁，而 Starlette 默认只发 ETag/Last-Modified、不发 Cache-Control，
    浏览器会按启发式缓存 index.html：按 F5 时直接吃缓存里的旧页面，再按旧页面里的
    版本号去加载旧 js，表现为「改了前端但页面没变」。加上 no-store 后普通刷新即可生效
    （?v=N 那套版本号仍然保留，浏览器禁用缓存时它只是冗余保险）。
    """

    async def get_response(self, path: str, scope):
        response = await super().get_response(path, scope)
        if response.status_code == 200:
            response.headers["Cache-Control"] = "no-store, must-revalidate"
        return response


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    common.logger.exception("未处理异常：%s %s", request.method, request.url.path)
    return JSONResponse(
        status_code=500,
        content={"detail": f"服务器内部错误（{type(exc).__name__}）：{exc}"},
    )


for routes in (status, media, history, detection, datasets, jobs, head_cuts, face_recognition, aggregate, actions):
    app.include_router(routes.router)

# Register the catch-all static mount after every API router.
common.STATIC_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/", NoCacheStaticFiles(directory=common.STATIC_DIR, html=True), name="static")

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
