"""The stored render of a photographed target, served behind the login.

These files are the only copy the system has (the Mac deletes a session once it
ships), and they are photographs of a range trip — so they are never on a static
mount. The route takes an integer primary key and reads the path from the row, so no
part of the URL ever reaches the filesystem.
"""

from fastapi import HTTPException, status
from fastapi.responses import FileResponse

from target_analyzer.api.deps import DbSession
from target_analyzer.api.endpoints.dashboard import dashboard_router
from target_analyzer.config import get_settings
from target_analyzer.models import Image

router = dashboard_router()


@router.get("/image/{image_id}")
def normalized_image(session: DbSession, image_id: int) -> FileResponse:
    image = session.get(Image, image_id)

    if image is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="no such image")

    root = get_settings().data_path.resolve()
    path = (root / image.normalized_path).resolve()

    # The URL carries only an integer, so this guards the column rather than the
    # request: it holds if a writer ever puts a traversing path in the row.
    if not path.is_relative_to(root) or not path.is_file():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="no such image")

    return FileResponse(
        path,
        # The render is always a PNG (the ingest endpoint refuses anything else);
        # image.content_type describes the original, which may be a JPEG.
        media_type="image/png",
        headers={"Cache-Control": "private, max-age=3600"},
    )
