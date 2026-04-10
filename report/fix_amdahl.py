with open('/Users/julia/Desktop/courses/c2_claude/gitlab/report/report.tex', 'r') as f:
    text = f.read()

# Define the text to move
amdahl_block = r"""\textbf{Amdahl analysis.}
We use Amdahl's law $S_{\text{Amdahl}}(T)=1/(s+(1-s)/T)$ to quantify the effective
serial fraction $s$~\cite{amdahl1967}. Per-point estimates from
$s=(T/S(T)-1)/(T-1)$:

\begin{table}[H]
\centering
\begin{tabular}{@{}llrrr@{}}
\toprule
Version & $n$ & $T$ & $S(T)$ & $s$ \\
\midrule
V2 & 500  & 16 & 3.02$\times$ & 28.7\% \\
V2 & 1000 & 16 & 7.98$\times$ & 6.7\% \\
V2 & 4000 & 64 & 37.1$\times$ & 1.2\% \\
V3 & 500  &  8 & 3.46$\times$ & 18.8\% \\
V3 & 1000 & 16 & 9.28$\times$ & 4.8\% \\
V3 & 4000 & 64 & 44.4$\times$ & 0.7\% \\
\bottomrule
\end{tabular}
\caption[Amdahl serial-fraction estimates from peak speedup]{Amdahl serial-fraction estimates from peak speedup.}
\label{tab:amdahl}
\end{table}

At small $n$, $s$ is large because $2n$ barriers and serial \texttt{omp single}
work are a significant fraction of total time; $S_{\infty}=1/s\approx3.5\times$
matches observed saturation. At $n=4000$, $s$ drops to $1.2\%$ as $O(n^{3})$ Schur
work dominates, enabling $37\times$ speedup. V3 reduces $s$ at every size by
removing column scaling from the serial region ($s$ drops from $1.2\%$ to $0.7\%$
at $n=4000$). Note that $s$ bundles explicit serial code, barrier waiting, and
bandwidth effects much larger than the naive $O(n)/O(n^{3})$ FLOP argument.
"""

# Find and remove the block from its current location
if amdahl_block in text:
    text = text.replace(amdahl_block, "")
    print("Found and removed amdahl block.")
else:
    print("Amdahl block not found exactly as specified.")

# Find insertion point: right after the 3rd enumerate item for regimes
insert_point = r"""    $+13\%$ gain from $T=32$ to $T=76$.
\end{enumerate}"""

if insert_point in text:
    text = text.replace(insert_point, insert_point + "\n\n" + amdahl_block)
    print("Successfully inserted amdahl block before figures.")
else:
    print("Insertion point not found.")

with open('/Users/julia/Desktop/courses/c2_claude/gitlab/report/report.tex', 'w') as f:
    f.write(text)
