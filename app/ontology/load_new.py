# app/ontology/loader.py
from dataclasses import dataclass
from typing import Dict, List, Optional
from rdflib import Graph, Namespace, URIRef, Literal

GM = Namespace("http://greenmind.ai/ontology#")   # đổi nếu TTL của bạn dùng IRI khác
OCEAN = ["O","C","E","A","N"]

@dataclass
class MetricDef:
    key: str                 # vd: "soda_units" hay "brand_diversity_norm"
    term: str                # IRI hoặc CURIE của term trong ontology
    unit: str                # "count", "bool", "0..1", "VND"...
    normalize: Optional[str] = None  # tên hàm chuẩn hoá (nếu có), để engine biết cách xử lý

@dataclass
class RuleWeight:
    metric_key: str
    weights: Dict[str, float]   # {"O":0.01,"C":-0.02,...}
    bias: float = 0.0           # nếu có

@dataclass
class PurchasingConfig:
    metrics: List[MetricDef]
    rules: List[RuleWeight]     # linear weights theo metric
    alpha: float = 0.1          # EMA smoothing mặc định

class OntologyLoader:
    def __init__(self, ttl_path: str):
        self.g = Graph()
        self.g.parse(ttl_path, format="turtle")

    def load_purchasing(self) -> PurchasingConfig:
        g = self.g
        metrics: List[MetricDef] = []
        rules: List[RuleWeight] = []

        # --- 1) Metrics ---
        # Mỗi metric trong TTL nên có các triple kiểu:
        #  ?m a gm:Metric ;
        #     gm:appliesToBehavior "purchasing" ;
        #     gm:metricKey "soda_units" ;
        #     gm:term gm:PURCH_SOFT_DRINK ;
        #     gm:unit "count" ;
        #     gm:normalize "minmax_0_10" .  (optional)
        for m in g.subjects(predicate=GM.appliesToBehavior, object=Literal("purchasing")):
            key = str(g.value(m, GM.metricKey))
            term = str(g.value(m, GM.term))
            unit = str(g.value(m, GM.unit) or "")
            norm = g.value(m, GM.normalize)
            metrics.append(MetricDef(key=key, term=term, unit=unit, normalize=str(norm) if norm else None))

        # --- 2) Linear weights / formulas ---
        # Mỗi rule trong TTL nên có dạng:
        #  ?r a gm:OceanRule ;
        #     gm:appliesToBehavior "purchasing" ;
        #     gm:metricKey "soda_units" ;
        #     gm:weight_O -0.006 ; gm:weight_C -0.006 ; gm:weight_E 0.0 ; gm:weight_A 0.0 ; gm:weight_N 0.003 ;
        #     gm:bias 0.0 .
        for r in g.subjects(predicate=GM.appliesToBehavior, object=Literal("purchasing")):
            if (r, GM.metricKey, None) in g and (r, GM.weight_O, None) in g:
                mk = str(g.value(r, GM.metricKey))
                w = {
                    "O": float(g.value(r, GM.weight_O) or 0.0),
                    "C": float(g.value(r, GM.weight_C) or 0.0),
                    "E": float(g.value(r, GM.weight_E) or 0.0),
                    "A": float(g.value(r, GM.weight_A) or 0.0),
                    "N": float(g.value(r, GM.weight_N) or 0.0),
                }
                bias = float(g.value(r, GM.bias) or 0.0)
                rules.append(RuleWeight(metric_key=mk, weights=w, bias=bias))

        # --- 3) Alpha (optional)
        a = g.value(URIRef(GM + "purchasing"), GM.alpha)
        alpha = float(a) if a else 0.1
        return PurchasingConfig(metrics=metrics, rules=rules, alpha=alpha)
