"""Offline corpus inference replay; unknown text uses explicitly labeled rules."""
import csv, json, re
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
def norm(t): return re.sub(r"[\W_]+", "", t)
class LocalPredictor:
    checkpoint = "corpus-inference-table"
    device = "cpu"
    def __init__(self):
        with (ROOT / "dataset/inference_full.csv").open(encoding="utf-8",newline="") as f: self.rows=list(csv.DictReader(f))
        self.meta={}
        for kind in ("dynasty","place","imagery","emotion"):
            self.meta[kind]={x["id"]:x.get("zh",x["id"]) for x in json.loads((ROOT/"labels"/(kind+"_labels.json")).read_text(encoding="utf-8"))["labels"]}
        self.place_meta={x["id"]:x for x in json.loads((ROOT/"labels/place_labels.json").read_text(encoding="utf-8"))["labels"]}
    def predict(self,text,imagery_threshold=0.5):
        key=norm(text)
        if not key: raise ValueError("请输入有效诗词")
        row=next((r for r in self.rows if key==norm(r["title"]) or key in norm(r["text"])),None)
        def label(kind, ident, confidence):
            return {"id":ident,"label":self.meta[kind].get(ident,ident),"confidence":float(confidence),"probabilities":{ident:float(confidence)}}
        if row:
            result={kind:label(kind,row["pred_"+kind],row["pred_"+kind+"_conf"]) for kind in ("dynasty","place","emotion")}
            imgs=[label("imagery",i,row["pred_imagery_conf"]) for i in row["pred_imagery"].split("|") if i]
            mode="corpus_inference_replay"
        else:
            matches=[(i,l) for i,l in self.meta["place"].items() if l in text]
            pid,pl= max(matches,key=lambda x:len(x[1])) if matches else ("PLC_UNKNOWN","未识别地名")
            result={"dynasty":label("dynasty","DYN_UNKNOWN",0),"place":label("place",pid,0.5 if matches else 0),"emotion":label("emotion","EMO_UNKNOWN",0)}
            imgs=[]; mode="rules_unclassified"
        result["imagery"]={"threshold":imagery_threshold,"labels":[x for x in imgs if x["confidence"]>=imagery_threshold],"confidence":max((x["confidence"] for x in imgs),default=0),"probabilities":{x["id"]:x["confidence"] for x in imgs}}
        result["confidence"]={k:result[k]["confidence"] for k in ("dynasty","place","emotion","imagery")}
        result.update(text=text,model={"checkpoint":self.checkpoint,"device":"cpu","mode":mode,"pretrained_model":"offline-corpus"})
        return result
