"""VLM captioning + yes/no 'is it exciting' verdict, with validated JSON and fallbacks."""
import json
import re

from PIL import Image

PROMPT = ('Look at this image. Reply with ONLY a JSON object with keys: '
          '"caption" (one sentence describing the scene), '
          '"exciting" ("yes" or "no": is something exciting or important happening?), '
          '"reason" (short reason for the answer).')
Q_CAPTION = "Describe this image in one short sentence."
Q_EXCITING = "Is something exciting or important happening in this image? Answer yes or no, with a short reason."


def _validate(d):
    cap = str(d["caption"]).strip()
    ans = str(d["exciting"]).strip().lower()
    if not cap or ans not in ("yes", "no"):
        raise ValueError("invalid field values")
    return {"caption": cap, "exciting": ans == "yes", "reason": str(d.get("reason", "")).strip()}


def extract_json(text):
    """Strict parse first; if the model never closed the object, salvage fields by regex."""
    m = re.search(r"\{.*\}", text, re.S)
    if m:
        try:
            return _validate(json.loads(m.group(0)))
        except Exception:
            pass
    cap = re.search(r'"caption"\s*:\s*"(.*?)"\s*(?:,|\}|$)', text, re.S)
    ans = re.search(r'"exciting"\s*:\s*"?\s*(yes|no)', text, re.I)
    rea = re.search(r'"reason"\s*:\s*"(.*?)(?:"\s*\}?\s*$|"\s*,|$)', text, re.S)
    if not (cap and ans):
        raise ValueError("could not find caption/exciting in output")
    return _validate({"caption": cap.group(1), "exciting": ans.group(1),
                      "reason": rea.group(1) if rea else ""})


class VLM:
    def __init__(self, model_id="HuggingFaceTB/SmolVLM-256M-Instruct"):
        import torch
        from transformers import AutoProcessor
        try:
            from transformers import AutoModelForImageTextToText as Auto
        except ImportError:
            from transformers import AutoModelForVision2Seq as Auto
        self.torch = torch
        self.processor = AutoProcessor.from_pretrained(model_id)
        # one 512px view per image instead of many tiles: much faster on CPU
        if hasattr(self.processor, "image_processor"):
            self.processor.image_processor.do_image_splitting = False
        self.model = Auto.from_pretrained(model_id, dtype=torch.float32).eval()

    def _gen(self, image, prompt, prefill="", max_new_tokens=100):
        msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
        text = self.processor.apply_chat_template(msgs, add_generation_prompt=True) + prefill
        inputs = self.processor(text=text, images=[image], return_tensors="pt")
        with self.torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        return self.processor.batch_decode(out[:, inputs["input_ids"].shape[1]:],
                                           skip_special_tokens=True)[0]

    def _guided(self, image):
        """Fill a fixed JSON skeleton field by field, so the structure is always valid."""
        pre = '{"caption": "'
        cap = self._gen(image, PROMPT, pre, 40).split('"')[0].strip()
        pre += cap + '", "exciting": "'
        ans = self._gen(image, PROMPT, pre, 3).strip().lower()
        ans = "yes" if ans.startswith("yes") else "no"
        pre += ans + '", "reason": "'
        rea = self._gen(image, PROMPT, pre, 40).split('"')[0].strip()
        return extract_json(json.dumps({"caption": cap, "exciting": ans, "reason": rea}))

    def analyze(self, image_path):
        image = Image.open(image_path).convert("RGB")
        # 1) single-shot JSON, prefilled with the opening of the object
        raw = ""
        try:
            raw = self._gen(image, PROMPT, '{"caption": "', 120)
            res = extract_json('{"caption": "' + raw)
            res["source"] = "json"
            return res
        except Exception as e:
            print(f"[vlm] single-shot JSON failed ({e}); raw={raw[:120]!r}")
        # 2) different method (not a repeat): guided field-by-field JSON
        try:
            res = self._guided(image)
            res["source"] = "json_guided"
            return res
        except Exception as e:
            print(f"[vlm] guided JSON failed: {e}")
        # 3) plain-text fallback
        try:
            cap = self._gen(image, Q_CAPTION, max_new_tokens=50).strip()
            ans = self._gen(image, Q_EXCITING, max_new_tokens=40)
            m = re.search(r"\b(yes|no)\b", ans.lower())
            return {"caption": cap or "No caption available", "exciting": bool(m and m[1] == "yes"),
                    "reason": ans.strip(), "source": "fallback_text"}
        except Exception as e:
            print(f"[vlm] fallback failed: {e}")
        return {"caption": "No caption available", "exciting": False,
                "reason": "VLM failed", "source": "default"}


def skip_result():
    return {"caption": "", "exciting": False, "reason": "VLM skipped", "source": "skipped"}