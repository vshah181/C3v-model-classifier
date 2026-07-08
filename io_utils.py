def write_nodes(unique_candidates, c__vals, a5_vals, fname="Nodes.dat"):
    with open(fname, "w") as f:
        f.write("  c'  |  a5  |       k (fractional)       | gap (eV) | chirality \n")
        f.write("-----------------------------------------------------------------\n")
        for (c__idx, a5_idx), info in sorted(unique_candidates.items()):
            c_ = c__vals[c__idx]
            a5 = a5_vals[a5_idx]
            k1, k2, k3 = info["k"]
            gap = info["gap"]
            charge = info["chirality"]
            f.write(f"{c_:6.3f}|{a5:6.3f}|({k1:8.5f},{k2:8.5f},{k3:8.5f})")
            f.write(f"| {gap:8.2E} | {charge:9.6f} \n")


def write_heatmap(c__vals, a5_vals, heatmap, fname="phase_diagram.csv"):
    with open(fname, "w") as f:
        f.write("#    c'    ,     a5     , phase\n")
        f.write("#------------------------------\n")
        for ic_, c_ in enumerate(c__vals):
            for ia5, a5 in enumerate(a5_vals):
                f.write(f"{c_:11.7f}, {a5:11.7f}, {heatmap[ic_, ia5]:4.1f}\n")
