import marimo

__generated_with = "0.23.6"
app = marimo.App()


@app.cell
def _():
    def kaprekar_step(n):
        # n is formatted as a 4 digit string with leading zeros
        s = f"{n:04d}"
        asc = int("".join(sorted(s)))
        desc = int("".join(sorted(s, reverse=True)))
        return desc - asc

    edges = set()
    for i in range(10000):
        edges.add((i, kaprekar_step(i)))

    nodes = set()
    for e in edges:
        nodes.add(e[0])
        nodes.add(e[1])

    graph_nodes = [{'id': str(n)} for n in nodes]
    graph_edges = [{'source': str(u), 'target': str(v)} for u, v in edges]
    return edges, graph_edges, graph_nodes, kaprekar_step, nodes


@app.cell
def _(graph_edges, graph_nodes):
    from wigglystuff import GraphWidget

    widget = GraphWidget()
    widget.nodes = graph_nodes
    widget.edges = graph_edges
    widget
    return GraphWidget, widget


if __name__ == "__main__":
    app.run()
