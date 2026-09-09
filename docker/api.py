from fastapi import FastAPI, UploadFile, File, BackgroundTasks
from fastapi.responses import FileResponse
import subprocess
import tempfile
import os
import shutil

app = FastAPI()

@app.post("/process")
def process_image(
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = BackgroundTasks(),
):
    tmpdir = tempfile.mkdtemp()

    input_path = os.path.join(tmpdir, file.filename)
    with open(input_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    subprocess.check_call([
        "python", "homr/main.py", input_path
    ])

    base, _ = os.path.splitext(input_path)
    output_path = base + ".musicxml"

    background_tasks.add_task(shutil.rmtree, tmpdir)

    return FileResponse(
        output_path,
        media_type="application/xml",
        filename=os.path.basename(output_path),
    )