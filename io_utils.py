def write_nodes(unique_candidates, r_vals, r__vals, fname="Nodes.dat"):
    with open(fname, "w") as f:
        f.write("  r   |  r'  |       k (fractional)       | gap (eV) | chirality \n")
        f.write("-----------------------------------------------------------------\n")
        for (r_idx, r__idx), info in sorted(unique_candidates.items()):
            r = r_vals[r_idx]
            r_ = r__vals[r__idx]
            k1, k2, k3 = info["k"]
            gap = info["gap"]
            charge = info["chirality"]
            f.write(f"{r:6.3f}|{r_:6.3f}|({k1:8.5f},{k2:8.5f},{k3:8.5f})")
            f.write(f"| {gap:8.3E} | {charge:9.6f} \n")

