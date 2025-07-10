# download.py
from pathlib import Path
import requests

def prepare_notebook_utils():
    if not Path("notebook_utils.py").exists():
        r = requests.get(
            url="https://raw.githubusercontent.com/openvinotoolkit/openvino_notebooks/latest/utils/notebook_utils.py"
        )
        open("notebook_utils.py", "w").write(r.text)

    # 이 부분은 utils가 있는 이후에 실행해야 가능
    import notebook_utils as utils
    return utils

def download_model(model_name="human-pose-estimation-0001", precision="FP16-INT8"):
    from notebook_utils import download_file
    base_model_dir = Path("model")
    model_path = base_model_dir / "intel" / model_name / precision / f"{model_name}.xml"

    if not model_path.exists():
        model_url_dir = (
            f"https://storage.openvinotoolkit.org/repositories/open_model_zoo/2022.1/models_bin/3/{model_name}/{precision}/"
        )
        download_file(model_url_dir + model_name + ".xml", model_path.name, model_path.parent)
        download_file(model_url_dir + model_name + ".bin", model_path.with_suffix(".bin").name, model_path.parent)

    return model_path
