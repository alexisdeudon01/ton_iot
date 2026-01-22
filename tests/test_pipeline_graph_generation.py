from src.app.pipeline.main import build_pipeline_graph

def test_pipeline_graph_generation():
    graph = build_pipeline_graph()
    mermaid = graph.to_mermaid()
    
    assert "graph TD" in mermaid
    assert "T00_InitRun" in mermaid
    assert "T17_Evaluate" in mermaid
    
    order = graph.get_execution_order()
    assert len(order) == 16
    assert order[0] == "T00_InitRun"
    assert order[-1] == "T17_Evaluate"
