def write_nodes(unique_candidates, c__vals, a12_vals, fname="Nodes.dat"):
    with open(fname, "w") as f:
        f.write("  c'  |  a12 |       k (fractional)       | gap (eV) | chirality \n")
        f.write("-----------------------------------------------------------------\n")
        for (c__idx, a12_idx), info in sorted(unique_candidates.items()):
            c_ = c__vals[c__idx]
            a12 = a12_vals[a12_idx]
            k1, k2, k3 = info["k"]
            gap = info["gap"]
            charge = info["chirality"]
            f.write(f"{c_:6.3f}|{a12:6.3f}|({k1:8.5f},{k2:8.5f},{k3:8.5f})")
            f.write(f"| {gap:8.2E} | {charge:9.6f} \n")


def write_heatmap(c__vals, a12_vals, heatmap, fname="phase_diagram.csv"):
    with open(fname, "w") as f:
        f.write("#    c'    ,    a12    , phase\n")
        f.write("#------------------------------\n")
        for ic_, c_ in enumerate(c__vals):
            for ia12, a12 in enumerate(a12_vals):
                f.write(f"{c_:11.7f}, {a12:11.7f}, {heatmap[ic_, ia12]:4.1f}\n")
