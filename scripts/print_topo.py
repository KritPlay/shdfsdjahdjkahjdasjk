from pathlib import Path

from rcpsp_marketing.io.psplib_extended import PSPLibExtendedParser
from rcpsp_marketing.io.psplib_base import PSPLibBaseParser
from rcpsp_marketing.core.precedence import topo_sort, is_topological_order, order_without_dummies, random_topo_sort


def main():
    # поменяй на нужный файл
    path = Path(r"data\extended\j30.sm\j301_1_with_metrics.sm")

    if "with_metrics" in path.name:
        proj = PSPLibExtendedParser().parse(path)
        print("[ok] parsed EXTENDED")
    else:
        proj = PSPLibBaseParser().parse(path)
        print("[ok] parsed RAW")

    for s in [1, 2, 3]:
        order = random_topo_sort(proj, seed=s)
        print(f"\nseed={s} ok={is_topological_order(proj, order)} len={len(order)}")
        print("full order:", order)
        print("real only :", order_without_dummies(proj, order)) # без фиктивных задач


if __name__ == "__main__":
    main()
