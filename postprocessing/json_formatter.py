import json
import logging
import numpy as np
from typing import Dict


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def create_json_ans(params: Dict, pages_count: int, full_text: str) -> json:
    json_dict = {
        "pages": pages_count,
        "text by pages": params,
        "full_text": full_text
    }
    json_ans = json.dumps(json_dict, ensure_ascii=False, indent="\t", separators=(',', ': '))
    logging.debug("json was created")
    return json_ans
