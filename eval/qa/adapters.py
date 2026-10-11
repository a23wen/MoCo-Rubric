"""Caption-only inputs and source-pinned benchmark scoring helpers."""
import ast
import hashlib
import importlib.util
import json
import re
from pathlib import Path
from typing import List, Optional

QASYSTEM = "You are given a caption describing an image, and a question about the image. Answer with a SINGLE LETTER (A, B, C, ...), no explanation."


class Adapters:
    def __init__(self, root):
        self.root = Path(root)
        manifest = json.loads(Path(__file__).with_name("sources.json").read_text())
        for name, info in manifest.items():
            p = self.root / name
            if not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest() != info["sha256"]:
                raise ValueError(f"Missing/changed scoring source {name}; run eval/qa/download_metrics.py")
        self.text = self.module("textvqa.py").TextVQAAccuracyEvaluator()
        self.doc = self.module("docvqa_challenge.py")
        self.blink = self.functions("blink_test.py", {"analyze_answer"})
        self.match = self.functions("blink_match.py", {"build_prompt"})
        tree = ast.parse((self.root / "blink_test.py").read_text())
        self.disclaimer = next(ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign)
                               and any(isinstance(t, ast.Name) and t.id == "disclaimer" for t in n.targets))
        tree = ast.parse((self.root / "internvl_vqa.py").read_text())
        self.short_prompt = next(ast.literal_eval(n.value) for n in ast.walk(tree) if isinstance(n, ast.Assign)
                                and any(isinstance(t, ast.Name) and t.id == "base_prompt" for t in n.targets))
        tree = ast.parse((self.root / "captionqa.py").read_text())
        ns = {"Optional": Optional, "List": List, "re": re}
        for n in tree.body:
            if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id in {"LETTER_ALPH", "CANNOT_ANSWER_TEXT"} for t in n.targets):
                exec(compile(ast.Module(body=[n], type_ignores=[]), "captionqa.py", "exec"), ns)
        self.captionqa = self.functions("captionqa.py", {"extract_letter", "normalize_gt_letter", "is_yesno_question",
                          "add_cannot_answer_option", "build_caption_qa_prompt"}, ns)

    def functions(self, file, names, namespace=None):
        ns = {} if namespace is None else namespace
        tree = ast.parse((self.root / file).read_text())
        nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
        if {n.name for n in nodes} != set(names):
            raise ValueError(f"Missing upstream functions in {file}")
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(self.root / file), "exec"), ns)
        return ns

    def module(self, file):
        spec = importlib.util.spec_from_file_location("moco_metric_" + file.replace('.', '_'), self.root / file)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def qa_input(self, q, captions):
        ds = q["bench"]
        if ds == "CaptionQA":
            ns = self.captionqa
            options = ns["add_cannot_answer_option"](q["question"], q["choices"])
            perm = q["permutation"]
            if sorted(perm) != list(range(len(options))):
                raise ValueError("CaptionQA requires the published option permutation")
            text = ns["build_caption_qa_prompt"](captions[q["image_id"]], q["question"], [options[i] for i in perm])
            messages = [{"role": "system", "content": QASYSTEM}, {"role": "user", "content": text}]
        else:
            context = "\n\n".join("Image " + str(i+1) + " caption:\n" + captions[k] for i, k in enumerate(q["image_ids"]))
            if ds == "BLINK":
                prefix = self.disclaimer if q["category"].replace(' ', '_') in ["Forensic_Detection", "Jigsaw", "Art_Style"] else ""
                prompt = prefix + q["official_prompt"] + "\nAnswer with the option letter only."
            else:
                if ds not in {"TextVQA", "DocVQA", "ChartQA"}:
                    raise ValueError(ds)
                prompt = q["question"] + " " + self.short_prompt
            messages = [{"role": "user", "content": context + "\n\n" + prompt}]
        return {"sample_id": q["sample_id"], "messages": messages}

    def captionqa_score(self, q, prediction):
        ns = self.captionqa
        letter = ns["extract_letter"](prediction, len(q["permutation"]))
        correct = cannot = False
        score = 0.0
        if letter is not None:
            idx = q["permutation"][ord(letter)-65]
            options = ns["add_cannot_answer_option"](q["question"], q["choices"])
            if options[idx] == ns["CANNOT_ANSWER_TEXT"]:
                cannot = True
                score = 1 / len(q["choices"]) + 0.05
            else:
                expected = ns["normalize_gt_letter"](q["choices"], q["answer"])
                if expected is None:
                    raise ValueError("Invalid CaptionQA reference answer")
                correct = idx == ord(expected)-65
                score = float(correct)
        return {"score": round(score, 4), "parsed_letter": letter, "is_correct": correct, "is_cannot_answer": cannot}

    def blink_choice(self, q, prediction):
        pending = []
        def callback(question, options, answer):
            pending.append(self.match["build_prompt"](question, options, answer))
            return None
        self.blink["match_multiple_choice"] = callback
        value = self.blink["analyze_answer"]({"question": q["question"], "choices": q["options"]}, prediction,
                                             ['('+chr(65+i)+')' for i in range(len(q["options"]))])
        return value, pending[0] if pending else None


def clean(text):
    text = re.sub(r'^\s*<think>\s*</think>\s*', '', text).strip()
    if '<think>' in text or '</think>' in text:
        raise ValueError("Nonempty thinking output")
    return text
